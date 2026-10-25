---
id: PB-ACT
title: Unauthorized actuator commands to pumps and valves
tags: [actuator, pump, valve, unauthorized command, write commands, manipulation of control, modbus, controller]
references:
  - MITRE ATT&CK for ICS T0855 Unauthorized Command Message
  - MITRE ATT&CK for ICS T0831 Manipulation of Control
  - MITRE ATT&CK for ICS T0836 Modify Parameter
  - IEC 62443-3-3 System security requirements and security levels
---
## Indicators
Pumps or valves change state against the control logic, for example a transfer pump running while the downstream tank is full or an inlet valve closed while the upstream level is low. A burst of write commands on the control network, writes from an unexpected source host, or commands outside maintenance windows are typical. Unauthorized command messages map to T0855 and their physical effect to T0831.

## Investigation steps
1. Identify the source host and protocol function codes of recent write commands to the affected controller and compare them with the approved engineering workstations.
2. Compare the actuator state with the interlock and control logic expected for the current process conditions.
3. Check the controller operating mode and whether its program or parameters changed recently.
4. Confirm with operations whether any manual intervention or maintenance was scheduled for the asset.
5. Review authentication and remote-access logs for the engineering workstation and jump hosts.

## Containment requiring approval
1. Coordinate with the shift supervisor to place the affected actuators under local or manual control so that the process stays within safe limits.
2. Block the unauthorized command source at the zone or conduit firewall only after operations confirms that legitimate control traffic is not affected.
3. Preserve the controller program and event logs before any program download or restart.

## Recovery
1. Verify controller logic against the known-good baseline before returning to automatic operation.
2. Rotate credentials used on the engineering workstation and review remote-access paths.
