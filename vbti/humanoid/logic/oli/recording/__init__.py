"""recording/ — Robot-side capture of the live camera streams (MAY-173 slam-demo-loop).

The World emits (CameraPublisher); this package ATTACHES A SINK: a standalone
recorder process drains the frame channel + the GT debug-pose channel into the
neutral coverage-dump layout (`mapping/recorder.DriveRecorder`), from which the
offline bake (rosbag synth → cuVSLAM) is re-runnable. Recording on the Robot
side is what transfers to the real robot: sensors stream; the robot listens.

Brain-pure (numpy + comm + camera_mounts); never imports isaacsim/limxsdk.
"""
