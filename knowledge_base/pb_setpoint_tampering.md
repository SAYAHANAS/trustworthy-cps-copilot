---
id: PB-SETPT
title: Set-point and parameter tampering in chemical dosing
tags: [setpoint, parameter, chemical dosing, ph, conductivity, water quality, analyzer, safety]
references:
  - MITRE ATT&CK for ICS T0836 Modify Parameter
  - MITRE ATT&CK for ICS T0880 Loss of Safety
  - MITRE ATT&CK for ICS T0838 Modify Alarm Settings
  - NIST SP 800-82 Guide to Operational Technology Security
---
## Indicators
A water-quality analyzer such as pH or conductivity drifts steadily away from its target although dosing pumps run normally. A short burst of write commands may precede the drift, and alarm limits may have been widened to hide it. Changing set-points maps to T0836; in chemical dosing the consequence can affect public health and safety (T0880).

## Investigation steps
1. Compare the active dosing set-points and alarm limits in the controller with the approved values in the configuration baseline.
2. Ask operations to take a grab sample and measure water quality with a laboratory instrument to confirm the analyzer reading.
3. Review write commands and engineering-workstation sessions around the start of the drift.
4. Estimate when the analyzer will cross regulatory or safety limits at the current drift rate and inform the shift supervisor.

## Containment requiring approval
1. With the shift supervisor and water-quality lead, restore approved set-points or switch dosing to manual control according to the standard operating procedure.
2. Hold or divert treated water if quality limits may be exceeded, following the plant safety procedure.
3. Preserve controller parameters and logs as evidence before restoring the baseline.
