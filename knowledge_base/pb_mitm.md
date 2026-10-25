---
id: PB-MITM
title: Adversary-in-the-middle on industrial protocols
tags: [man in the middle, adversary in the middle, arp spoofing, intercept, reporting message, integrity, network]
references:
  - MITRE ATT&CK for ICS T0830 Adversary-in-the-Middle
  - MITRE ATT&CK Enterprise T1557 Adversary-in-the-Middle
  - MITRE ATT&CK for ICS T0856 Spoof Reporting Message
---
## Indicators
Address-resolution changes for controllers or HMIs, duplicate hardware addresses, unexpected latency on control traffic, or reporting messages whose values do not match field measurements. Interception lets an adversary alter both what operators see and what controllers receive (T0830).

## Investigation steps
1. Compare current address-resolution tables of the HMI and controllers with the documented network baseline.
2. Capture traffic on the affected segment and check for a host relaying control traffic between the HMI and controllers.
3. Verify critical process values through local panels or redundant instruments.

## Containment requiring approval
1. Remove the intercepting host from the segment after operations confirms that control traffic will not be disrupted.
2. Configure static address entries or port security for critical controllers during a planned change window.
