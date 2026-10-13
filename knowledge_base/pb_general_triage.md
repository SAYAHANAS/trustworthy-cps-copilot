---
id: PB-TRIAGE
title: General triage of CPS security alerts
tags: [triage, incident handling, safety first, operations, escalation, evidence preservation, monitoring]
references:
  - NIST SP 800-61 Computer Security Incident Handling Guide
  - NIST SP 800-82 Guide to Operational Technology Security
  - IEC 62443-2-1 Security program requirements for IACS asset owners
---
## Principles
In operational technology, physical safety and process continuity come before containment. Security staff propose actions; operations staff decide how the process is run. Automatic blocking, isolation or shutdown of control equipment can itself cause a hazardous event and must never happen without explicit human approval.

## Investigation steps
1. Confirm the alert by checking the underlying signals and whether independent sources agree.
2. Determine which process stage and asset are affected and their safety and production criticality.
3. Inform the operations shift lead and agree on how the process will be monitored during the investigation.
4. Preserve logs, network captures and controller state before any restart or configuration change.
5. Escalate to the incident response lead when a safety-relevant asset is involved or the evidence indicates active manipulation.

## Monitoring guidance
Low-risk events are recorded for trend analysis. Repeated low-risk events on the same asset, or a rising predicted probability of attack over successive horizons, justify closer review even when no single event crosses a threshold.
