# #!/usr/bin/env bash
# set -e

# # Example rosbag record script.
# # Modify topics in configs/topics.yaml first.

# ros2 bag record \
#   /camera_top/color/image_raw \
#   /camera_wrist_left/color/image_raw \
#   /camera_wrist_right/color/image_raw \
#   /joint_states \
#   /left_gripper/state \
#   /right_gripper/state \
#   /left_arm_controller/command \
#   /right_arm_controller/command \
#   /left_gripper/command \
#   /right_gripper/command


# 注意：这个脚本只录 ROS2 反馈和控制。三路 Python 相机后面需要单独写一个 record_python_cameras.py 或者在主节点里同步保存 JPEG/MP4。



# chmod +x ~/LingBotVLA/lingbotvla_deploy/scripts/record.sh
#!/usr/bin/env bash
set -euo pipefail

# Record ROS2 topics for LingBot-VLA deployment.
#
# Cameras are NOT recorded here because this project captures cameras by Python:
#   - left_wrist_rgb RealSense
#   - right_wrist_rgb RealSense
#   - head_rgb OpenCV
#
# This script records only ROS2 topics from:
#   configs/topics.yaml -> ros_topics.states
#   configs/topics.yaml -> ros_topics.commands
#   configs/topics.yaml -> ros_topics.tf
#
# Usage:
#   cd ~/LingBotVLA/lingbotvla_deploy
#   bash scripts/record.sh
#
# Optional:
#   bash scripts/record.sh --name test_pick
#   bash scripts/record.sh --config configs/topics.yaml
#   bash scripts/record.sh --output-dir bags
#   bash scripts/record.sh --no-commands
#   bash scripts/record.sh --no-tf
#   bash scripts/record.sh --dry-run

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONFIG="${PROJECT_ROOT}/configs/topics.yaml"
OUTPUT_DIR="${PROJECT_ROOT}/bags"
BAG_NAME="lingbotvla_$(date +%Y%m%d_%H%M%S)"

RECORD_STATES=1
RECORD_COMMANDS=1
RECORD_TF=1
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --name)
      BAG_NAME="$2"
      shift 2
      ;;
    --no-states)
      RECORD_STATES=0
      shift
      ;;
    --no-commands)
      RECORD_COMMANDS=0
      shift
      ;;
    --no-tf)
      RECORD_TF=0
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      sed -n '1,40p' "$0"
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown argument: $1"
      exit 1
      ;;
  esac
done

if [[ ! -f "$CONFIG" ]]; then
  echo "[ERROR] Config file not found: $CONFIG"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

TOPICS="$(
python3 - "$CONFIG" "$RECORD_STATES" "$RECORD_COMMANDS" "$RECORD_TF" <<'PY'
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("[ERROR] Python package 'yaml' not found. Install with: python3 -m pip install pyyaml", file=sys.stderr)
    sys.exit(1)

config_path = Path(sys.argv[1]).expanduser().resolve()
record_states = sys.argv[2] == "1"
record_commands = sys.argv[3] == "1"
record_tf = sys.argv[4] == "1"

with open(config_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

ros_topics = cfg.get("ros_topics", {})

topics = []

if record_states:
    for _, topic in ros_topics.get("states", {}).items():
        topics.append(topic)

if record_commands:
    for _, topic in ros_topics.get("commands", {}).items():
        topics.append(topic)

if record_tf:
    for _, topic in ros_topics.get("tf", {}).items():
        topics.append(topic)

# Remove duplicates while preserving order.
seen = set()
unique_topics = []
for topic in topics:
    if topic and topic not in seen:
        seen.add(topic)
        unique_topics.append(topic)

for topic in unique_topics:
    print(topic)
PY
)"

if [[ -z "$TOPICS" ]]; then
  echo "[ERROR] No topics found in config: $CONFIG"
  exit 1
fi

mapfile -t TOPIC_ARRAY <<< "$TOPICS"

echo "======================================================================"
echo "LingBot-VLA ROS2 Bag Record"
echo "======================================================================"
echo "Project root : $PROJECT_ROOT"
echo "Config       : $CONFIG"
echo "Output dir   : $OUTPUT_DIR"
echo "Bag name     : $BAG_NAME"
echo "Record states   : $RECORD_STATES"
echo "Record commands : $RECORD_COMMANDS"
echo "Record TF       : $RECORD_TF"
echo "----------------------------------------------------------------------"
echo "Topics:"
for topic in "${TOPIC_ARRAY[@]}"; do
  echo "  $topic"
done
echo "======================================================================"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[DRY RUN] Command would be:"
  echo "ros2 bag record -o \"$OUTPUT_DIR/$BAG_NAME\" ${TOPIC_ARRAY[*]}"
  exit 0
fi

ros2 bag record \
  -o "$OUTPUT_DIR/$BAG_NAME" \
  "${TOPIC_ARRAY[@]}"


