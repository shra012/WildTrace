# xArm 7 assets

`xarm_ros2_source/` is a sparse checkout of UFACTORY's official
`xArm-Developer/xarm_ros2` `humble` branch at commit
`62936f7ea1846a85f7350de2c4c18f39e6d19715`. Only the description package and
repository root metadata were checked out. No xArm driver is used.

`xarm7_isaac.urdf.xacro` wraps only the official geometric/dynamic xArm 7 macro.
It intentionally omits ROS 2 control, Gazebo, the UFACTORY SDK, hardware plugins,
grippers, and IP parameters. It adds a fixed 120 mm simulation stylus and an
explicit `pen_tip` frame below the official `link_eef` flange frame.

Generate `xarm7_with_pen.urdf` with:

```bat
cd /d C:\wildtrace\isaac_xarm7_drawing
set "PYTHONEXE=C:\Users\Aswin\isaac-sim-standalone-6.0.1-windows-x86_64\kit\python\python.exe"
"C:\Users\Aswin\isaac-sim-standalone-6.0.1-windows-x86_64\python.bat" scripts\generate_xarm7_urdf.py
```

The generator uses the project-local `xacro_vendor` package. To recreate that
dependency without a global install:

```bat
"C:\Users\Aswin\isaac-sim-standalone-6.0.1-windows-x86_64\python.bat" -m pip install --target assets\xarm7\xacro_vendor xacro==2.1.1
```

`xarm7.usd` is intentionally not fabricated. `check_environment.py` imports the
generated URDF through the installed Isaac Sim URDF importer and writes the USD.
