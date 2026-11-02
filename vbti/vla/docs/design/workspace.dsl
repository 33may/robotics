workspace "VBTI Modular Robot Learning Platform" "C4 architecture model focused on modular reuse across 3D reconstruction and AI model workflows." {
    !identifiers hierarchical

    model {
        operator = person "Human Operator" "Collects demonstrations, prepares assets, launches training/evaluation, and monitors robot behaviour."

        rawMedia = softwareSystem "Raw Capture Sources" "Videos, photos, depth captures, and object scans used as reconstruction input." "External"
        hf = softwareSystem "Hugging Face Hub" "Stores LeRobot datasets and trained policy checkpoints." "External"
        isaac = softwareSystem "Isaac Sim / Isaac Lab" "Simulation runtime, scene rendering, synthetic data generation, and generated task execution." "External"
        cosmos = softwareSystem "Cosmos Transfer" "Photorealistic video augmentation for sim-to-real data variation." "External"

        robotHardware = softwareSystem "SO-ARM101 Robot Workcell" "Physical robot arm, leader arm, RealSense cameras, table scene, and task objects." {
            followerArm = container "Follower Arm" "Executes policy actions on the real SO-ARM101 robot." "Feetech servos"
            leaderArm = container "Leader Arm" "Teleoperation input device for demonstration collection." "Feetech servos"
            cameras = container "RealSense Cameras" "Multi-view RGB/depth capture for observations and calibration." "Intel RealSense D405"
            taskScene = container "Task Scene" "Cup, duck, table, and workspace objects used for training/evaluation tasks."
        }

        vbti = softwareSystem "VBTI Modular Robot Learning Platform" "Reusable project platform for reconstructing task environments and training/evaluating robot-learning policies." {
            group "3D Reconstruction Pipeline" {
                reconstructionCli = container "Reconstruction Orchestrator" "CLI entry point that chains video processing, COLMAP, GS/MILo reconstruction, USD export, scene composition, and IsaacLab export." "logic/reconstruct/master.py"
                mediaProcessing = container "Media Processing Utilities" "Extracts sharp frames from raw videos/photos, fixes phone rotation metadata, and prepares image folders for reconstruction." "logic/reconstruct/video_utils.py"
                sfmReconstruction = container "SfM / Camera Reconstruction" "Runs Nerfstudio/COLMAP, validates sparse models, selects the best model, and undistorts images." "logic/reconstruct/colmap_utils.py"
                gsMeshReconstruction = container "Gaussian Splat + Mesh Reconstruction" "Trains Gaussian splats with MILo and extracts meshes through learnable SDF or Poisson repair scripts." "logic/reconstruct/gs_milo_utils.py, scripts/3d/"
                assetExport = container "3D Format + Physics Export" "Converts splats, point clouds, GLB, PLY, meshes, USD/USDA, collision meshes, and deformable assets for simulation use." "logic/reconstruct/format_utils.py, clean_mesh.py, scripts/3d/"
                sceneGeneration = container "Scene + Simulation Asset Generation" "Extracts scene config and generates LeIsaac/IsaacLab scene assets, task boilerplate, environment configs, cameras, lights, and robot placement." "logic/reconstruct/isaac_cfg_utils.py, robot_utils.py"
                cosmosPrep = container "Cosmos Augmentation Preparation" "Extracts HDF5 camera modalities, prepares RGB/depth/edge/seg videos, and writes Cosmos transfer configs." "logic/reconstruct/cosmos_transfer.py"
            }

            group "AI Model Pipeline" {
                hardwareUtils = container "Hardware Utilities" "Camera discovery/viewing/calibration/reset plus servo scanning, calibration, profiles, rest/unlock, and raw robot setup helpers." "logic/cameras/, logic/servos/"
                datasetUtils = container "Dataset Utilities" "Converts, trims, augments, subsamples, inspects, replays, validates, and edits LeRobot/HDF5 datasets." "logic/dataset/"
                perceptionUtils = container "Perception + Detection Utilities" "Runs object detection, phase detection, async detection, dataset processing, and distilled detector training/export." "logic/detection/, logic/inference/async_detector.py"
                depthUtils = container "Depth Utilities" "Bakes, colorizes, compares, captures, estimates, and adds gripper/depth features to datasets and realtime inference inputs." "logic/depth/, dataset depth tools"
                trainingUtils = container "Training Utilities" "Configures, launches, monitors, chains, and remotely runs SmolVLA/GR00T training backends and experiment utilities." "logic/train/"
                inferenceUtils = container "Inference Utilities" "Runs real policy inference, prompt/voice input, async chunk execution, policy loading, and closed-loop action dispatch." "logic/inference/"
                evaluationUtils = container "Evaluation Utilities" "Defines protocols, checkpoint sweeps, rendering, trial helpers, eval engine, and generated evaluation scenarios." "logic/inference/eval_*, protocols/"
                simulationUtils = container "Simulation Utilities" "Runs simulation playgrounds, generated Isaac/LeIsaac tasks, synthetic data paths, and domain-randomized environments." "scripts/sim/, generated IsaacLab/LeIsaac configs"
                remoteExecution = container "Remote Execution Utilities" "Synchronizes code/data/checkpoints and launches training/evaluation jobs on the robot or training machine." "logic/train/remote.py, remote.yaml"
                knowledgeBase = container "Design Knowledge Base" "Generated codegraph, August memory, module docs, process docs, and design DSL used to ground future architecture work." ".august/knowledge, .august/memory, docs/"
            }

            operator -> reconstructionCli "Starts reconstruction and scene-generation workflows"
            operator -> hardwareUtils "Uses for setup, calibration, collection, and troubleshooting"
            operator -> trainingUtils "Starts training experiments"
            operator -> evaluationUtils "Runs evaluation protocols"
            operator -> inferenceUtils "Runs policy deployment/evaluation"

            rawMedia -> mediaProcessing "Provides videos/photos/depth captures"
            reconstructionCli -> mediaProcessing "Extract frames and normalize media"
            reconstructionCli -> sfmReconstruction "Recover cameras and sparse geometry"
            reconstructionCli -> gsMeshReconstruction "Train splats and extract meshes"
            reconstructionCli -> assetExport "Convert assets and add physics/collision"
            reconstructionCli -> sceneGeneration "Compose simulator scenes and tasks"
            reconstructionCli -> cosmosPrep "Prepare augmentation controls"

            mediaProcessing -> sfmReconstruction "Prepared frames"
            sfmReconstruction -> gsMeshReconstruction "Undistorted images and COLMAP cameras"
            gsMeshReconstruction -> assetExport "PLY meshes, point clouds, splats, GLB assets"
            assetExport -> sceneGeneration "USD/USDA assets and collision meshes"
            sceneGeneration -> isaac "Generates runnable scenes/tasks for"
            cosmosPrep -> cosmos "Sends control videos/configs to"
            cosmosPrep -> datasetUtils "Consumes HDF5 episodes from"

            hardwareUtils -> robotHardware.cameras "Configures and reads"
            hardwareUtils -> robotHardware.leaderArm "Reads teleoperation input from"
            hardwareUtils -> robotHardware.followerArm "Calibrates and commands"
            hardwareUtils -> robotHardware.taskScene "Supports task setup for"
            hardwareUtils -> datasetUtils "Provides recorded episodes to"

            datasetUtils -> hardwareUtils "Consumes recorded observations/actions"
            datasetUtils -> perceptionUtils "Can add object/phase annotations through"
            datasetUtils -> depthUtils "Can add or transform depth features through"
            datasetUtils -> trainingUtils "Provides LeRobot datasets to"
            datasetUtils -> hf "Publishes and loads datasets from"
            perceptionUtils -> inferenceUtils "Provides realtime detections/state augmentation"
            depthUtils -> inferenceUtils "Provides depth features for realtime inputs"
            trainingUtils -> datasetUtils "Trains on prepared datasets"
            trainingUtils -> hf "Pulls/pushes datasets and checkpoints"
            inferenceUtils -> trainingUtils "Loads trained checkpoints from"
            inferenceUtils -> hardwareUtils "Sends policy actions and reads observations through"
            evaluationUtils -> inferenceUtils "Runs policy trials through"
            evaluationUtils -> hardwareUtils "Uses workcell setup and cameras through"
            simulationUtils -> sceneGeneration "Uses generated environments from"
            simulationUtils -> isaac "Runs simulated tasks in"
            remoteExecution -> trainingUtils "Launches remote training"
            remoteExecution -> evaluationUtils "Launches remote evaluation"
            remoteExecution -> hf "Syncs artifacts with"
            knowledgeBase -> reconstructionCli "Documents 3D pipeline entry points"
            knowledgeBase -> datasetUtils "Documents dataset formats and process gotchas"
            knowledgeBase -> trainingUtils "Documents experiments and decisions"
        }

        devMachine = deploymentEnvironment "Local Development Machine" {
            localNode = deploymentNode "Fedora Workstation" "Developer machine" "Fedora Linux, RTX GPU" {
                containerInstance vbti.reconstructionCli
                containerInstance vbti.mediaProcessing
                containerInstance vbti.sfmReconstruction
                containerInstance vbti.gsMeshReconstruction
                containerInstance vbti.assetExport
                containerInstance vbti.sceneGeneration
                containerInstance vbti.cosmosPrep
                containerInstance vbti.datasetUtils
                containerInstance vbti.trainingUtils
                containerInstance vbti.simulationUtils
                containerInstance vbti.knowledgeBase
            }
        }

        robotMachine = deploymentEnvironment "Robot / Remote Machine" {
            remoteNode = deploymentNode "Robot PC" "Machine attached to robot workcell" "Linux, USB cameras/servos" {
                containerInstance vbti.hardwareUtils
                containerInstance vbti.perceptionUtils
                containerInstance vbti.depthUtils
                containerInstance vbti.inferenceUtils
                containerInstance vbti.evaluationUtils
                containerInstance vbti.remoteExecution
            }

            workcellNode = deploymentNode "Physical Workcell" "Robot hardware and task setup" {
                containerInstance robotHardware.followerArm
                containerInstance robotHardware.leaderArm
                containerInstance robotHardware.cameras
                containerInstance robotHardware.taskScene
            }
        }
    }

    views {
        systemContext vbti "SystemContext" {
            include *
            autoLayout lr
            description "The modular VBTI platform in its operational context."
        }

        container vbti "Containers" {
            include *
            autoLayout lr
            description "The platform split into two reusable product layers: 3D reconstruction and AI model pipeline utilities."
        }

        dynamic vbti "ReconstructionFlow" "Raw media to simulator-ready scene." {
            rawMedia -> vbti.mediaProcessing "Videos/photos"
            vbti.mediaProcessing -> vbti.sfmReconstruction "Frames"
            vbti.sfmReconstruction -> vbti.gsMeshReconstruction "COLMAP cameras/images"
            vbti.gsMeshReconstruction -> vbti.assetExport "Splats/meshes/PLY"
            vbti.assetExport -> vbti.sceneGeneration "USD/collision assets"
            vbti.sceneGeneration -> isaac "Runnable task/env"
            autoLayout lr
        }

        dynamic vbti "ModelFlow" "Dataset to trained and evaluated robot policy." {
            operator -> vbti.hardwareUtils "Collect/setup"
            vbti.hardwareUtils -> vbti.datasetUtils "Recorded episodes"
            vbti.datasetUtils -> vbti.trainingUtils "LeRobot dataset"
            vbti.trainingUtils -> hf "Checkpoint"
            vbti.evaluationUtils -> vbti.inferenceUtils "Trial protocol"
            vbti.inferenceUtils -> vbti.hardwareUtils "Closed-loop actions"
            autoLayout lr
        }

        deployment * "Local Development Machine" "LocalDevelopmentDeployment" {
            include *
            autoLayout lr
            description "Project containers running on the local development workstation."
        }

        deployment * "Robot / Remote Machine" "RobotRemoteDeployment" {
            include *
            autoLayout lr
            description "Project containers running on the robot/remote machine and connected physical workcell."
        }

        styles {
            element "Person" {
                shape person
                background #174A7C
                color #ffffff
            }
            element "Software System" {
                background #116466
                color #ffffff
            }
            element "External" {
                background #6c757d
                color #ffffff
            }
            element "Container" {
                background #2C7A7B
                color #ffffff
            }
            element "Group" {
                color #ffffff
                stroke #4fd1c5
            }
            element "Deployment Node" {
                background #f8f9fa
                color #212529
            }
            relationship "Relationship" {
                color #adb5bd
            }
        }

        theme default
    }
}
