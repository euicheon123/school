import os
import xml.etree.ElementTree
import zipfile
import re

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


def extract_smali(apk, target_strings):
  smali_dir = os.path.join(apk, 'smali')
  smali_contents = ""

  for root, dirs, files in os.walk(smali_dir):
    for file in files:
      with open(os.path.join(root, file), 'r', encoding='utf-8', errors='ignore') as f:
        smali_code = f.read()
        for target_string in target_strings:
          if target_string in smali_code:
            smali_contents += smali_code + '\n'
            break

  return smali_contents

def is_res_encrypted(apk):
  res_dir = os.path.join(apk, 'res')
  for root, dirs, files in os.walk(res_dir):
    for file in files:
      with open(os.path.join(root, file), 'rb') as f:
        data = f.read()
        if re.search(b'encryption_marker', data):
          return True
  return False

def main():
  target_smali_strings = ["SmsManager", "Contact Resolver", "HttpURLConnection", "DefaultHttpClient", "AndroidHttpClient"]

  print("\nBENIGN - permission")
  for fname in os.listdir(BENIGN_DIR):
    apk = os.path.join(BENIGN_DIR, fname)
    print( extract_perm(apk) )
  
  print("\nMALWARE - permission")
  for fname in os.listdir(MALWARE_DIR):
    apk = os.path.join(MALWARE_DIR, fname)
    print( extract_perm(apk) )

  print("\nBENIGN - Smali")
  for fname in os.listdir(BENIGN_DIR):
    apk = os.path.join(BENIGN_DIR, fname)
    print("Smali code related to target strings:")
    print(extract_smali(apk, target_smali_strings))

  print("\nMALWARE - Smali")
  for fname in os.listdir(MALWARE_DIR):
    apk = os.path.join(MALWARE_DIR, fname)
    print("Smali code related to target strings:")
    print(extract_smali(apk, target_smali_strings))

  print("\nBENIGN - Res Encryption")
  for fname in os.listdir(BENIGN_DIR):
    apk = os.path.join(BENIGN_DIR, fname)
    print(f"Res folder encrypted: {is_res_encrypted(apk)}")

  print("\nMALWARE - Res Encryption")
  for fname in os.listdir(MALWARE_DIR):
    apk = os.path.join(MALWARE_DIR, fname)
    print(f"Res folder encrypted: {is_res_encrypted(apk)}")

if __name__ == "__main__":
  main()
