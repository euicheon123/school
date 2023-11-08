import os
import xml.etree.ElementTree

BENIGN_DIR = '/mnt/ssd/dku-apk/dku20_decoded/benign'
MALWARE_DIR = '/mnt/ssd/dku-apk/dku20_decoded/malware'

def extract_perm(apk):
  filepath = os.path.join(apk, 'AndroidManifest.xml')
  str = ""
  tree = xml.etree.ElementTree.parse(filepath)
  root = tree.getroot()
  for ele in root.iter("uses-permission"):
    for k in ele.attrib.keys():
      if 'name' in k:
        str += ele.attrib[k] + ' '
  return str
  
def extract_intent(apk):
  filepath = os.path.join(apk, 'AndroidManifest.xml')
  str = ""
  tree = xml.etree.ElementTree.parse(filepath)
  root = tree.getroot()
  for ele in root.iter("intent-filter"):
    for node in ele.findall('action'):
      for k in node.attrib.keys():
        if 'name' in k:
          str += node.attrib[k] + ' '
  return str
  
def main():

  print("\nBENIGN - permission")
  for fname in os.listdir(BENIGN_DIR):
    apk = os.path.join(BENIGN_DIR, fname)
    print( extract_perm(apk) )
  
  print("\nMALWARE - permission")
  for fname in os.listdir(MALWARE_DIR):
    apk = os.path.join(MALWARE_DIR, fname)
    print( extract_perm(apk) )
    
  print("\nBENIGN - intent")
  for fname in os.listdir(BENIGN_DIR):
    apk = os.path.join(BENIGN_DIR, fname)
    print( extract_intent(apk) )
    
  print("\nMALWARE - intent")
  for fname in os.listdir(MALWARE_DIR):
    apk = os.path.join(MALWARE_DIR, fname)
    print( extract_intent(apk) )
    
if __name__ == "__main__":
  main()
