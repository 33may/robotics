"""humanoid.logic.oli.reason.teleoperation — operator-driven reasoning.

Teleoperation is the family of reasoning where a human operator supplies the intent.
Each teleop *method* is one way the operator's signal enters the brain (joystick today;
gloves / mocopi / VR exist in the LimX vendor and would slot in here later). Every method
ends at the same `→ PolicyIn` contract (D5), so swapping methods never touches Action.

Current method:
- `joystick/` — gamepad/joystick teleop: a source produces stick axes, `JoystickAdapter`
  maps them to a body-frame velocity, and `Teleop` combines that with the held mode.
"""
