---
id: PB-CRED
title: Credential attacks against OT remote access and engineering systems
tags: [authentication failures, brute force, credentials, password, valid accounts, remote access, engineering workstation]
references:
  - MITRE ATT&CK Enterprise T1110 Brute Force
  - MITRE ATT&CK for ICS T0859 Valid Accounts
  - MITRE ATT&CK for ICS T0812 Default Credentials
  - IEC 62443-3-3 System security requirements and security levels
---
## Indicators
Repeated authentication failures against HMIs, engineering workstations, jump hosts or controllers, logins at unusual times, or use of vendor default accounts. A successful login that follows many failures is especially important because it may give the adversary valid access (T0859).

## Investigation steps
1. Identify the accounts and source addresses involved in the failed authentications and check whether any later login succeeded.
2. Check whether default or shared vendor accounts are enabled on the targeted systems.
3. Review sessions opened by the affected accounts and the commands or configuration changes made during them.

## Containment requiring approval
1. Disable or reset compromised accounts after confirming with operations that no critical process depends on an active session.
2. Restrict remote access to the targeted systems to approved jump hosts.
