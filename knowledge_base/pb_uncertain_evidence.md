---
id: PB-UNCERT
title: Handling uncertain or conflicting evidence
tags: [uncertain, conflicting evidence, low confidence, verify, sensor health, maintenance, out of distribution, abstain]
references:
  - NIST SP 800-82 Guide to Operational Technology Security
  - NIST SP 800-61 Computer Security Incident Handling Guide
---
## When evidence conflicts
Detection models, unsupervised deviation scores and predictions can disagree because of sensor faults, maintenance activities, seasonal operating modes or behaviour that was never seen during training. Conflicting evidence lowers confidence and should be resolved by verification rather than by acting on any single signal.

## Investigation steps
1. Check the maintenance and change calendar for work on the affected asset or network segment.
2. Verify sensor health, including calibration dates, diagnostics and communication errors of the involved instruments.
3. Compare the current operating mode, such as start-up, shutdown or cleaning cycles, with the modes present in the training period.
4. Collect additional observations over the next minutes and review whether the signals converge.

## Escalation
If verification cannot explain the discrepancy and the affected asset is safety-relevant, escalate to the incident response lead and the operations shift lead together.
