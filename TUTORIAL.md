# PatientGUI Tutorial

- [PatientGUI Tutorial](#patientgui-tutorial)
  - [Step 1a: Starting the Application](#step-1a-starting-the-application)
  - [Step 1b: Starting the Application (Simulated)](#step-1b-starting-the-application-simulated)
  - [Step 2: Config Mode Overview](#step-2-config-mode-overview)
  - [Step 3: Configure Your Sequence](#step-3-configure-your-sequence)
    - [Trial Metadata](#trial-metadata)
    - [Timing Settings](#timing-settings)
  - [Step 4: Add Movements to Sequence](#step-4-add-movements-to-sequence)
    - [Saving/Loading Configurations](#savingloading-configurations)
  - [Step 5: Switch to Patient Mode](#step-5-switch-to-patient-mode)
  - [Step 6: Movement Tutorial](#step-6-movement-tutorial)
  - [Step 7: Practice the Movement](#step-7-practice-the-movement)
  - [Step 8: Recording in Progress](#step-8-recording-in-progress)
  - [Step 9: Next Movement](#step-9-next-movement)
  - [Step 10: Recording Saved](#step-10-recording-saved)
  - [Step 0: Set Up Ripple](#step-0-set-up-ripple)
  - [Quick Reference](#quick-reference)



## Step 1a: Starting the Application

If this is your first time using the EMG system, ask Thien (or me, if for some reason I ams till in the lab) to first set it up for you
If none of us is availiable, please check [Step 0: Set Up Ripple](#step-0-set-up-ripple)

Connect the HD-sEMG Ripple system to the computer via ethernet cable

IMPORTANT: Make sure the Trellis application is open, running and connected to the device


Run `start.bat` to launch both the backend server and frontend application. Two terminal windows will open:

- **Backend Server**: Handles EMG data streaming and recording
- **Frontend Dev Server**: Serves the web interface

If everything went well, the two terminals should look something like this

Notice the "client connected" line at the bottom of the left terminal


![Startup](images/1-startup.png)

Once both servers are running, open your browser to `http://localhost:8080`.

---


## Step 1b: Starting the Application (Simulated)

Run `start_sim.bat` to launch both the backend server and frontend application. Two terminal windows will open:

- **Backend Server**: Handles EMG data streaming and recording
- **Frontend Dev Server**: Serves the web interface

If everything went well, the two terminals should look something like this

Notice the "client connected" line at the bottom of the left terminal


![Startup](images/1-startup.png)

Once both servers are running, open your browser to `http://localhost:8080`.

---

## Step 2: Config Mode Overview

When the application loads, you'll be in **CONFIG MODE**. This is where you set up your recording session.

![Config Mode](images/2-config_mode.png)

The interface consists of:
- **3D Hand Model** (center): Preview movements in real-time
- **Movement Selector** (right panel): Choose which movement to preview
- **Playback Controls** (bottom): Play, pause, and adjust animation speed
- **Sequence Controls** (right panel): Configure and start recording sessions

---

## Step 3: Configure Your Sequence

Click **"Configure Sequence"** to open the Sequence Configuration dialog.

![Sequence Configuration](images/3-sequenceconfig.png)

### Trial Metadata
- **Subject ID**: Enter the patient/subject identifier
- **Session ID**: Enter a session name (e.g., "session0")
- **Notes**: Add any additional notes about the recording

### Timing Settings
- **Rest Between Reps**: Pause duration between repetitions (seconds)
- **Prep Time**: Countdown before each exercise begins (seconds after rest, to give a heads up to the participant)
- **Pause Between Items**: Rest period between different movements (seconds)
- **Playback Speed**: Animation speed multiplier

---

## Step 4: Add Movements to Sequence

Scroll down in the configuration dialog to the **SEQUENCE ITEMS** section.

![Movement Configuration](images/4-movementconfig.png)

For each movement:
1. Select the movement type from the dropdown (e.g., "Ring + Pinky Flexion", "Wrist Flexion", "Tripod Pinch")
2. Set **Anim**: Animation variant (if available)
3. Set **Reps**: Number of repetitions for this movement
4. Use the arrow buttons to reorder movements
5. Click the red **X** to remove a movement
6. Click **"+ Add Item"** to add more movements

### Saving/Loading Configurations
- **Export JSON**: Save your current configuration to a file
- **Import JSON**: Load a previously saved configuration

**Important**: Click **Apply** when finished.

---

## Step 5: Switch to Patient Mode

After configuring your sequence, click **"Patient Mode"** in the right panel. The interface will switch to a simplified view designed for patients.

![Patient Mode](images/5-patientmode.png)

The **EXERCISE PROGRESS** panel shows all movements in the sequence with their completion status.

Click **"Start Exercise"** to begin the recording session.

---

## Step 6: Movement Tutorial

Before each movement, a tutorial screen appears explaining what the patient should do.

![Tutorial Start](images/6-tutorialstart.png)

The dialog shows:
- The movement name (e.g., "Ring + Pinky Flexion")
- Number of repetitions
- Instructions to watch and replicate the movement
- A reminder to notify staff if feeling discomfort

Click **"Start Tutorial"** to proceed.

---

## Step 7: Practice the Movement

The 3D hand model demonstrates the movement. You should help the participant (not patient otherwise Anna kills me):
1. Watch the animation carefully
2. Practice the movement
3. Drag to rotate the view if needed

![Tutorial](images/7-tutorial.png)

When the patient feels confident, click **"I'm Ready"** to begin recording.

---

## Step 8: Recording in Progress

During recording, EMG data is captured from all channels. You can view the live EMG data by clicking **"View EMG Data"** in the bottom right.

![EMG Channels](images/8-emg.png)

The EMG viewer shows:
- Real-time channels for each channel (for now only the last second of signal)
- Recording status indicator (green "RECORDING" badge)
- Channel labels (CH 1 through CH 8, etc.)

The patient performs the movement for the specified number of repetitions while data is recorded.

**Important:** Make sure the **Recording** badge is presnet and the waveforms appear **RED**

---

## Step 9: Next Movement

After completing all repetitions of a movement, the next movement in the sequence is presented.

![Next Movement](images/9-nextmovement.png)

The process repeats:
1. Tutorial screen appears
2. Patient watches demonstration
3. Patient clicks "I'm Ready"
4. Recording captures the movement
5. Continue until all movements are complete

---

## Step 10: Recording Saved

When the session ends, either naturally or by pressing the red **STOP** button on the bottom of the participant panel, the backend server saves the recording automatically.

Note the line at the end of the Temrinal

![Saved Recording](images/10-SavedRecording.png)

Recordings are saved to the `recordings/` folder with filenames containing:
- Timestamp (e.g., `20260205_154641`)
- Duration and channel count

Example: `emg_recording_20260205_154641.pkl (196.8s, 32 channels)`

---

## Step 0: Set Up Ripple

Streaming the data to the frontend requirest the Trellis application to first be set up and running 

You can download the latest version from [here](https://rippleneuro.com/trellis-eeg-software/)

Once it downloaded and installed you need to set up the connection: doing so requires an ethernet port, which is "sacrificed" to permit connection to the Ripple (If you need the ethernet cable for connection, use a dongle) 

To set up, on Windows

1. Go to Control Panel\Network and Internet\Network Connections 
NOTE: this is translated literally from italian, might not be the same 
2. Find your Ethernet port ![Setup](images/0-network-setup.png)
3. Right click on it, select properties
4. Uncheck all boxes, except the Ipv4 one
5. Click one the Ipv4 Box, then on the properties button below the list
6. Set the ip to static, insert the address and subnet maks shown in the picture

![Ethernet Setup](images/0b-port-setup.png)
Great artistic capabilities i know


1. Connect the Ripple via Ethernet cable 
2. Go on The trellis app (Only go here after you connected the system)
3. It should be automatically connected: if not change network mode to TCP, and insert the same ip as above, then click apply
4.  If it still does not connect, try the good ol' plug and unplug a couple of times, some days the Trellis gods just don't smile favorably upon you

## Quick Reference

| Action | How To |
|--------|--------|
| Preview a movement | Select from dropdown in Config Mode, click Play |
| Configure session | Click "Configure Sequence" |
| Start recording | Switch to Patient Mode, click "Start Exercise" |
| View live EMG | Click "View EMG Data" during recording |
| Stop session | Click "Stop" in the Exercise Progress panel |
| Find recordings | Check the `recordings/` folder in the backend directory |