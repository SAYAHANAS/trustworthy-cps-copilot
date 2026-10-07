---
id: PB-DOS
title: Denial of service and flooding on the control network
tags: [network, flooding, denial of service, packets, traffic volume, stale values, loss of view, availability]
references:
  - MITRE ATT&CK for ICS T0814 Denial of Service
  - MITRE ATT&CK for ICS T0815 Denial of View
  - MITRE ATT&CK Enterprise T1498 Network Denial of Service
  - NIST SP 800-82 Guide to Operational Technology Security
---
## Indicators
Control-network packet volume rises far above its normal level and polling of field devices degrades. Historian and HMI values stop changing because polls time out, so stale readings show near-zero variance. Operators may lose visibility of the process even though it keeps running. These behaviours map to T0814 and T0815.

## Investigation steps
1. Identify the top talkers and destination hosts responsible for the traffic increase on the control network.
2. Check which sensor values have become stale by comparing their recent variance with normal behaviour.
3. Confirm with operations whether the process is still under control and whether local panels show values that differ from the HMI.
4. Check switch and firewall health, including port utilisation and dropped packets, on the affected segment.

## Containment requiring approval
1. Rate-limit or block the flooding source at the segment boundary after operations confirms that essential control traffic will continue to flow.
2. Assign operators to local panels for critical measurements while remote visibility is degraded.

## Recovery
1. Restore normal polling and verify that historian values update with expected variance before relying on them again.
