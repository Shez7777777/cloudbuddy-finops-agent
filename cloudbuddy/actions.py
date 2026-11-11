"""Approval queue + executors.

The agent can only *request* a change. Someone has to approve it before an
executor runs, and the boto3 executor uses DryRun=True by default.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Protocol

VALID_ACTIONS = {"stop", "downsize"}


@dataclass
class ActionRequest:
    id: int
    instance_id: str
    action: str
    target_type: str | None
    reason: str
    status: str = "pending"   # pending -> rejected | executed | failed
    result: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))


class Executor(Protocol):
    def stop(self, instance_id: str) -> str: ...
    def resize(self, instance_id: str, target_type: str) -> str: ...


class MockExecutor:
    def __init__(self):
        self.log: list[str] = []

    def stop(self, instance_id):
        self.log.append(f"stop {instance_id}")
        return f"[mock] stopped {instance_id}"

    def resize(self, instance_id, target_type):
        self.log.append(f"resize {instance_id} -> {target_type}")
        return f"[mock] {instance_id}: stopped, changed to {target_type}, started"


class Boto3Executor:
    def __init__(self, region="us-east-1", dry_run=True):
        import boto3
        self.ec2 = boto3.client("ec2", region_name=region)
        self.dry_run = dry_run

    def _dry(self, fn, **kwargs):
        from botocore.exceptions import ClientError
        try:
            fn(DryRun=True, **kwargs)
        except ClientError as e:
            if e.response["Error"]["Code"] == "DryRunOperation":
                return "dry run ok - you have permission, nothing changed"
            raise
        return "dry run ok"

    def stop(self, instance_id):
        if self.dry_run:
            return self._dry(self.ec2.stop_instances, InstanceIds=[instance_id])
        self.ec2.stop_instances(InstanceIds=[instance_id])
        return "stopped"

    def resize(self, instance_id, target_type):
        if self.dry_run:
            return self._dry(self.ec2.modify_instance_attribute, InstanceId=instance_id,
                             InstanceType={"Value": target_type})
        # instance type can only be changed while stopped
        self.ec2.stop_instances(InstanceIds=[instance_id])
        self.ec2.get_waiter("instance_stopped").wait(InstanceIds=[instance_id])
        self.ec2.modify_instance_attribute(InstanceId=instance_id, InstanceType={"Value": target_type})
        self.ec2.start_instances(InstanceIds=[instance_id])
        return f"resized to {target_type}"


class ApprovalQueue:
    def __init__(self, executor: Executor):
        self.executor = executor
        self.requests: dict[int, ActionRequest] = {}
        self._ids = itertools.count(1)

    def submit(self, instance_id, action, reason, target_type=None) -> ActionRequest:
        if action not in VALID_ACTIONS:
            raise ValueError(f"action must be one of {sorted(VALID_ACTIONS)}")
        if action == "downsize" and not target_type:
            raise ValueError("downsize needs a target_type")
        # one open request per instance is enough
        for r in self.requests.values():
            if r.instance_id == instance_id and r.status == "pending":
                return r
        req = ActionRequest(next(self._ids), instance_id, action, target_type, reason)
        self.requests[req.id] = req
        return req

    def pending(self):
        return [r for r in self.requests.values() if r.status == "pending"]

    def reject(self, request_id):
        req = self._pending(request_id)
        req.status = "rejected"
        return req

    def approve(self, request_id):
        req = self._pending(request_id)
        try:
            if req.action == "stop":
                req.result = self.executor.stop(req.instance_id)
            else:
                req.result = self.executor.resize(req.instance_id, req.target_type)
            req.status = "executed"
        except Exception as e:
            req.status, req.result = "failed", str(e)
        return req

    def _pending(self, request_id):
        req = self.requests.get(request_id)
        if req is None or req.status != "pending":
            raise ValueError(f"no pending request with id {request_id}")
        return req
