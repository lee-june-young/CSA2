# Lab-SA Secure Protocols
import random, hashlib
from Crypto.Protocol.SecretSharing import Shamir
from Crypto.Cipher import AES
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP
from Crypto.Util import number

g = 2
#p = int( # 1536-bit MODP Group
#    'FFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD129024E088A67CC74020BBEA63B139B22514A08798E3404DDEF9519B3CD3A431B302B0A6DF25F14374FE1356D6D51C245E485B576625E7EC6F44C42E9A637ED6B0BFF5CB6F406B7EDEE386BFB5A899FA5AE9F24117C4B1FE649286651ECE45B3DC2007CB8A163BF0598DA48361C55D39A69163FA8FD24CF5F83655D23DCA3AD961C62F356208552BB9ED529077096966D670C354E4ABC9804F1746C08CA237327FFFFFFFFFFFFFFFF',
#    base=16
#)
p = 24379 # 15-bit

# find prime number
def primesInRange(x, y):
    prime_list = []
    for n in range(x, y):
        isPrime = True

        for num in range(2, n):
            if n % num == 0:
                isPrime = False

        if isPrime:
            prime_list.append(n)
    return prime_list

# key generate
def generateKeyPair(g, p):
    pri_key = random.randrange(50000, 90000) #temp
    pub_key = (g ** pri_key) % p
    return pub_key, pri_key

# key agreement with hash function (md5)
# return 128bit key
def agree(sk, pk, p):
    # key = H((g^a)^b)
    key = (pk ** sk) % p
    # key agreement composed with a hash function md5: generate 128-bit key
    return hashlib.md5(bytes(key)).hexdigest()

# encrypt using AES-128
# return encrypted hex string
def encrypt(key, plaintext):
    encryptor = AES.new(key=bytes.fromhex(key), mode=AES.MODE_CBC, iv=bytes([0x00]*16))
    boundary = 16 # Data must be padded to 16 byte boundary in CBC mode
    pad = lambda s: s + (boundary - len(s) % boundary) * chr(boundary - len(s) % boundary)
    raw = pad(plaintext)
    return encryptor.encrypt(bytes(raw, encoding='utf-8')).hex()

# decrypt using AES-128
# return plaintext
def decrypt(key, ciphertext):
    decryptor = AES.new(key=bytes.fromhex(key), mode=AES.MODE_CBC, iv=bytes([0x00]*16))
    unpad = lambda s: s[0:-ord(s[-1])] # unpad since we add padding in encryption (16bit-block)
    decrypted = decryptor.decrypt(bytes.fromhex(ciphertext)).decode('utf-8')
    return unpad(decrypted)

# Return a random N-bit prime number
def getPrime(bit):
    return number.getPrime(bit)

# Return a generator of modulo
def getOneGenerator(p):
    required_set = set(num for num in range (1, p) if number.GCD(num, p) == 1)
    for g in range(1, p):
        actual_set = set(pow(g, powers) % p for powers in range (1, p))
        if required_set == actual_set:
            return g

def getOneGenerator2(p):
    for g in range(2, p):
        flag = True
        for q in range(2, p):
            if number.isPrime(q) and ((p-1) % q) == 0:
                # q is prime factor
                if g ** int((p-1) / q) % p == 1:
                    flag = False
                    break
        if flag:
            return g
    return None

# Secret-Sharing
def make_shares(key, t, n):
    return Shamir.split(t, n, key)

def combine_shares(shares):
    key = Shamir.combine(shares)
    return int.from_bytes(key, 'big')

# RSA
def generateRSAKeyPair():
    key = RSA.generate(1028)
    publicKey = PKCS1_OAEP.new(key.publickey())
    privateKey = PKCS1_OAEP.new(key)
    return (publicKey, privateKey)

def encryptRSA(publicKey, plainText):
    return publicKey.encrypt(plainText.encode())

def decryptRSA(privateKey, cipherText):
    return privateKey.decrypt(cipherText).decode()
