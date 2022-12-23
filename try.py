import time
import random
from qqdm import qqdm, format_str

tw = qqdm(range(10), desc=format_str('bold', 'Description'))

a = {2:1,3:2,4:4}
print({3:1,
        **a})