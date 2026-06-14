# GRADIS Litchi Operator Card - mini2_seed_patrol

- command_id: seed
- action: SEED_PATROL
- source: local
- reason: 
- csv: 1781324665_mini2_seed_patrol.csv

## Operator Steps
1. Open Litchi Mission Hub or Litchi on the Android controller.
2. Import the CSV if one was generated.
3. Verify Mini 2, altitude, RTH altitude, geofence, finish action, and battery.
4. Keep RC link active for the entire mission. Mini 2 Litchi waypoint flight uses virtual-stick style control, so link loss must trigger RTH.
5. Start or reject the action as pilot in command.

## Notes
- This is the default demo patrol around the configured home point.
- CSV stores waypoint rows only; set Litchi global mission settings manually after import.
