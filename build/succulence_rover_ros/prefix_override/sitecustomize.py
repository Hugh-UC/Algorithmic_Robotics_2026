import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/hugh/algorithmic-robots-world/workspace/succulence_ws/install'
