import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/workspace/succulence_rover_ros/install/succulence_rover_ros'
