import hashlib


def GetHashKey(text: str):
    """ 
    forget why to encrypt user phone information
    將用戶的資料加密
    
    """
    hash_object = hashlib.sha256()
    hash_object.update(text.encode())
    return hash_object.hexdigest()

