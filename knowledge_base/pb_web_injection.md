---
id: PB-WEB
title: Injection and cross-site scripting against OT web interfaces
tags: [injection, cross site scripting, xss, http, web interface, public facing, exploitation, dns]
references:
  - MITRE ATT&CK for ICS T0819 Exploit Public-Facing Application
  - MITRE ATT&CK for ICS T0866 Exploitation of Remote Services
  - MITRE ATT&CK Enterprise T1190 Exploit Public-Facing Application
---
## Indicators
Unusual HTTP request bodies, script fragments or database syntax in requests to HMI web servers, IoT gateways or historians, error responses in bursts, and new accounts or files created on the web host afterwards. Exploiting exposed services is a common initial access path into OT networks (T0819, T0866).

## Investigation steps
1. Review web server logs for the suspicious requests and the responses returned to them.
2. Check whether the targeted interface is reachable from outside the OT zone and why.
3. Look for new accounts, scheduled tasks or files created on the web host after the requests.

## Containment requiring approval
1. Restrict access to the vulnerable interface to approved management hosts after operations confirms it is not needed for live control.
2. Apply the vendor patch or mitigation during a planned change window.
