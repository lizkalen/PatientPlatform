# Tenoexo Bluetooth API Documentation

## Overview

The tenoexo exposes a Bluetooth RFCOMM SPP (Serial Port Profile) server. Communication uses **JSON** messages over UTF-8 encoded strings. The server advertises service UUID `ba1291e2-f751-4df3-9511-b9fddd33e97e` ("TenoexoServer").

---

## Connection

| Property | Value |
|----------|-------|
| Protocol | RFCOMM (SPP) |
| Service UUID | `ba1291e2-f751-4df3-9511-b9fddd33e97e` |
| Service Name | TenoexoServer |
| Encoding | UTF-8 |
| Message Format | JSON |

**Discovery:** Use `bluetooth.find_service(uuid="ba1291e2-f751-4df3-9511-b9fddd33e97e")` to discover the tenoexo. Connect to `(host, port)` from the first match.

---

## Request Message Structure

Every **request** from client to tenoexo must be a JSON object:

```json
{
  "command": "<method_name>",
  "send": <unique_id>,
  "args": [<arg1>, <arg2>, ...]
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `command` | string | **Yes** | Name of the API method to call |
| `send` | number | **Yes** | Unique message ID (used for acknowledgment) |
| `args` | array | **Yes** | List of arguments for the command. Use `[]` for no-arg commands |

### Acknowledgment Response

For every request that includes `"send"`, the tenoexo replies with:

```json
{
  "received": <same_id_as_send>
}
```

### Push Messages (Server → Client)

The tenoexo also sends unsolicited status updates when values change. These are JSON objects with a single key:

```json
{ "<status_key>": <value> }
```

---

## Commands Reference

### Hand Control

#### `trigger_open_close`
Toggle hand state: opens if closed, closes if open.

```json
{"command": "trigger_open_close", "send": 1, "args": []}
```

---

### Emergency & Safety

#### `emergency_stop`
Activate or release app-triggered emergency stop. When active, motors are disabled.

| Arg | Type | Description |
|-----|------|-------------|
| 0 | bool | `true` = emergency stop pressed, `false` = released |

```json
{"command": "emergency_stop", "send": 2, "args": [true]}
{"command": "emergency_stop", "send": 3, "args": [false]}
```

---

### Grip Modes

#### `key_grip`
Enable or disable key grip (lateral pinch) mode.

| Arg | Type | Description |
|-----|------|-------------|
| 0 | bool | `true` = enable, `false` = disable |

```json
{"command": "key_grip", "send": 4, "args": [true]}
```

#### `midway_opening`
Enable or disable midway opening mode.

| Arg | Type | Description |
|-----|------|-------------|
| 0 | bool | `true` = enable, `false` = disable |

```json
{"command": "midway_opening", "send": 5, "args": [true]}
```

---

### Force & Speed Parameters

All factor parameters are **float** in range `[0, 1]`. `0` = minimum, `1` = maximum.

#### `closing_force`
Closing force scale (0–1).

```json
{"command": "closing_force", "send": 6, "args": [0.75]}
```

#### `finger_opening_force`
Finger opening force scale (0–1).

```json
{"command": "finger_opening_force", "send": 7, "args": [0.5]}
```

#### `thumb_opening_force`
Thumb opening force scale (0–1).

```json
{"command": "thumb_opening_force", "send": 8, "args": [0.5]}
```

#### `action_delay`
Action delay factor (0 = minimal, 1 = maximal).

```json
{"command": "action_delay", "send": 9, "args": [0.5]}
```

#### `finger_speed`
Finger speed factor (0–1).

```json
{"command": "finger_speed", "send": 10, "args": [0.5]}
```

#### `thumb_speed`
Thumb speed factor (0–1).

```json
{"command": "thumb_speed", "send": 11, "args": [0.5]}
```

---

### Therapy / IMU

#### `trigger_therapy`
Start or stop therapy exercise.

| Arg | Type | Description |
|-----|------|-------------|
| 0 | bool | `true` = start, `false` = stop |
| 1 | int | Exercise number |

```json
{"command": "trigger_therapy", "send": 12, "args": [true, 1]}
{"command": "trigger_therapy", "send": 13, "args": [false, 0]}
```

#### `pause_therapy`
Pause or resume therapy recording.

| Arg | Type | Description |
|-----|------|-------------|
| 0 | bool | `true` = pause, `false` = resume |

```json
{"command": "pause_therapy", "send": 14, "args": [true]}
```

#### `evaluate_therapy`
Submit therapy evaluation after completion.

| Arg | Type | Description |
|-----|------|-------------|
| 0 | int | Therapy evaluation (e.g. rating) |
| 1 | bool | Whether stopped due to emergency |

```json
{"command": "evaluate_therapy", "send": 15, "args": [5, false]}
```

---

### Status Requests (Client-initiated)

These commands request current status. The tenoexo replies with the corresponding push message(s).

#### `request_hand_status`
Get current hand state.

```json
{"command": "request_hand_status", "send": 20, "args": []}
```

Response: `{"hand_status": <int>}` — see [TenoexoState](#tenoexostate) values.

#### `request_key_grip_status`
Get key grip mode.

```json
{"command": "request_key_grip_status", "send": 21, "args": []}
```

Response: `{"key_grip_status": <bool>}`

#### `request_midway_opening_status`
Get midway opening mode.

```json
{"command": "request_midway_opening_status", "send": 22, "args": []}
```

Response: `{"midway_opening_status": <bool>}`

#### `request_battery_status`
Get battery state.

```json
{"command": "request_battery_status", "send": 23, "args": []}
```

Response: `{"battery_status": <int>}` — see [BatteryState](#batterystate) values.

#### `request_all_statuses`
Request all statuses. The tenoexo sends multiple push messages and ends with `sync_status: 2`.

```json
{"command": "request_all_statuses", "send": 24, "args": []}
```

#### `request_emergency_stop_status`
Get emergency stop state.

```json
{"command": "request_emergency_stop_status", "send": 25, "args": []}
```

Response: `{"emergency_stop_status": <bool>}`

#### `request_therapy_status`
Get current therapy status.

```json
{"command": "request_therapy_status", "send": 26, "args": []}
```

Response: `{"therapy_status": <int>}` — see [TherapyStatus](#therapystatus) values.

#### `request_cycle_counter_value`
Get grip cycle count for current therapy.

```json
{"command": "request_cycle_counter_value", "send": 27, "args": []}
```

Response: `{"cycle_counter_value": <int>}`

#### `request_initial_hand_status`
Get initial hand state for therapy.

```json
{"command": "request_initial_hand_status", "send": 28, "args": []}
```

Response: `{"initial_hand_status": <int>}` — TenoexoState value.

#### `request_therapy_duration_value`
Get current therapy duration in seconds.

```json
{"command": "request_therapy_duration_value", "send": 29, "args": []}
```

Response: `{"therapy_duration_value": <int>}`

#### `request_closing_force_value`
Get closing force factor.

```json
{"command": "request_closing_force_value", "send": 30, "args": []}
```

Response: `{"closing_force_value": <float>}`

#### `request_action_delay_value`
Get action delay factor.

```json
{"command": "request_action_delay_value", "send": 31, "args": []}
```

Response: `{"action_delay_value": <float>}`

#### `request_finger_speed_value`
Get finger speed factor.

```json
{"command": "request_finger_speed_value", "send": 32, "args": []}
```

Response: `{"finger_speed_value": <float>}`

#### `request_thumb_speed_value`
Get thumb speed factor.

```json
{"command": "request_thumb_speed_value", "send": 33, "args": []}
```

Response: `{"thumb_speed_value": <float>}`

#### `request_finger_opening_force_value`
Get finger opening force factor.

```json
{"command": "request_finger_opening_force_value", "send": 34, "args": []}
```

Response: `{"finger_opening_force_value": <float>}`

#### `request_thumb_opening_force_value`
Get thumb opening force factor.

```json
{"command": "request_thumb_opening_force_value", "send": 35, "args": []}
```

Response: `{"thumb_opening_force_value": <float>}`

---

### Input Devices

#### `input_devices`
Get list of active input device names.

```json
{"command": "input_devices", "send": 40, "args": []}
```

Response: `{"input_devices": [<str>, ...]}`

---

### Shoulder IMU (if active)

#### `calibrate_shoulder_imu`
Start shoulder IMU calibration.

| Arg | Type | Description |
|-----|------|-------------|
| 0 | int | Calibration duration in seconds |

```json
{"command": "calibrate_shoulder_imu", "send": 50, "args": [10]}
```

#### `cancel_calibrate_shoulder_imu`
Cancel ongoing shoulder IMU calibration.

```json
{"command": "cancel_calibrate_shoulder_imu", "send": 51, "args": []}
```

Response: `{"calibration_success": <int>}` — sent when calibration ends (0 = fail, 1 = success, etc.).

---

### System

#### `set_time`
Set Raspberry Pi system time (for log timestamps).

| Arg | Type | Description |
|-----|------|-------------|
| 0 | int | Year |
| 1 | int | Month (1–12) |
| 2 | int | Day |
| 3 | int | Hour (0–23) |
| 4 | int | Minute |
| 5 | int | Second |

```json
{"command": "set_time", "send": 60, "args": [2025, 2, 12, 14, 30, 0]}
```

#### `heart_beat`
Connection keep-alive. No effect, use to verify link.

```json
{"command": "heart_beat", "send": 61, "args": []}
```

---

## Push Message Keys (Server → Client)

| Key | Type | Description |
|-----|------|-------------|
| `hand_status` | int | TenoexoState |
| `key_grip_status` | bool | Key grip mode |
| `midway_opening_status` | bool | Midway opening mode |
| `battery_status` | int | BatteryState |
| `closing_force_value` | float | Closing force factor |
| `action_delay_value` | float | Action delay factor |
| `finger_speed_value` | float | Finger speed factor |
| `thumb_speed_value` | float | Thumb speed factor |
| `finger_opening_force_value` | float | Finger opening force factor |
| `thumb_opening_force_value` | float | Thumb opening force factor |
| `therapy_status` | int | TherapyStatus |
| `emergency_stop_status` | bool | Emergency stop active |
| `cycle_counter_value` | int | Grip cycle count |
| `initial_hand_status` | int | Initial TenoexoState for therapy |
| `therapy_duration_value` | int | Therapy duration (seconds) |
| `input_devices` | list[str] | Active input device names |
| `sync_status` | int | `2` = all statuses sent (after `request_all_statuses`) |
| `calibration_success` | int | Shoulder IMU calibration result |
| `recording_warning` | int | Recording issue warning |
| `received` | any | Acknowledgment (matches `send` id) |

---

## Enumerations

### TenoexoState

| Value | Name | Description |
|-------|------|-------------|
| 0 | Open | Hand fully open |
| 1 | Closed | Hand fully closed |
| 2 | Opening | In motion: opening |
| 3 | Closing | In motion: closing |
| 4 | Semi_Closed | Partially closed |

### BatteryState

| Value | Name |
|-------|------|
| 0 | Full |
| 1 | Medium |
| 2 | Low |
| 3 | Empty |

### TherapyStatus

| Value | Name |
|-------|------|
| 0 | STANDBY |
| 1 | STARTING |
| 2 | RECORDING |
| 3 | SAVING |
| 4 | PAUSED |
| 5 | EVALUATING |

---

## Example Session

```
Client → Server: {"command": "trigger_open_close", "send": 1, "args": []}
Server → Client: {"received": 1}
Server → Client: {"hand_status": 2}     # Opening (state changed)
Server → Client: {"hand_status": 0}     # Open (motion finished)

Client → Server: {"command": "request_battery_status", "send": 2, "args": []}
Server → Client: {"received": 2}
Server → Client: {"battery_status": 0}
```

---

## Notes

1. **Message boundaries:** Each JSON object is sent as one UTF-8 string. Multiple objects may be concatenated; parsing should handle this (e.g. split by `}\s*{` or read line-by-line if newline-delimited).
2. **Command execution:** A command is only executed if both `command` and `send` are present. The `args` array is always required; use `[]` when there are no arguments.
3. **App trigger:** The `trigger_open_close` command requires AppTrigger to be enabled in `input_devices.yaml` on the tenoexo.
4. **App communication thread:** Bluetooth handling must be started (uncomment `app_connection_thread` in `tenoexo.py`).
