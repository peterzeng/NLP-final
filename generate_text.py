import sys
import xml.etree.ElementTree as ET

# Generates body from raw
tree = ET.parse('./Oliver Twist/raw.xml')
sys.stdout = open('book.txt', 'w')
root = tree.getroot()

for child in root:
    if child.tag == "body":
        ET.dump(child)