from pathlib import Path
from typing import Union, List, Optional

PATH = Union[Path, str]
STR_LIST_STR = Optional[Union[str, List[str]]]

SURVEY_URLS = {
    2006: "https://www.ine.gub.uy/c/document_library/get_file?uuid=1e2b8c68-9a06-4d2c-9bff-c8e5a67c5d43&groupId=10181",
    2007: "https://www.ine.gub.uy/c/document_library/get_file?uuid=ac9034f4-7cf8-40fa-b0bf-9834d679ea5f&groupId=10181",
    2008: "https://www.ine.gub.uy/c/document_library/get_file?uuid=991de40a-1040-44ed-9f19-3a1278d0acf6&groupId=10181",
    2009: "https://www.ine.gub.uy/c/document_library/get_file?uuid=cc730465-7c74-4898-8344-cbb5b74d8aa2&groupId=10181",
    2010: "https://www.ine.gub.uy/c/document_library/get_file?uuid=9cb93aa0-af7b-4b53-9229-e30e98288ea6&groupId=10181",
    2011: "https://www.ine.gub.uy/c/document_library/get_file?uuid=cc986929-5916-4d4f-a87b-3fb20a169879&groupId=10181",
    2012: "https://www.ine.gub.uy/c/document_library/get_file?uuid=144daa3d-0ebf-4106-ae11-a150511addf9&groupId=10181",
    2013: "https://www.ine.gub.uy/c/document_library/get_file?uuid=9ddf38cc-99bb-4196-992b-77530b025237&groupId=10181",
    2014: "https://www.ine.gub.uy/c/document_library/get_file?uuid=68cc1d11-e017-4a6d-a749-5a1e1a4a5306&groupId=10181",
    2015: "https://www.ine.gub.uy/c/document_library/get_file?uuid=7c62ef78-0cc6-4fba-aae4-921ff5ceddd6&groupId=10181",
    2016: "https://www.ine.gub.uy/c/document_library/get_file?uuid=715c873b-539f-4e92-9159-d38063270951&groupId=10181",
    2017: "https://www.ine.gub.uy/c/document_library/get_file?uuid=e38ea53c-7253-4007-9f67-2f5f161eea91&groupId=10181",
    2018: "https://www.ine.gub.uy/c/document_library/get_file?uuid=b63b566f-8d11-443d-bcd8-944f137c5aaf&groupId=10181",
    2019: "https://www.ine.gub.uy/c/document_library/get_file?uuid=8c934d2a-ad67-4208-8f21-96989696510e&groupId=10181",
    2020: "https://www.ine.gub.uy/c/document_library/get_file?uuid=17e8cbb6-85dc-46e3-9567-8d196553125f&groupId=10181",
}

DICTIONARY_URLS = {
    2006: "https://www.ine.gub.uy/c/document_library/get_file?uuid=423d4ef6-b46b-4fd6-89f2-9406069a6626&groupId=10181",
    2019: "https://www.ine.gub.uy/c/document_library/get_file?uuid=800e3c63-5cbc-4842-ad00-745f801f9220&groupId=10181",
    2020: "https://www.ine.gub.uy/c/document_library/get_file?uuid=1f65ca4f-3d97-40bc-bf7b-56b5de811359&groupId=10181",
}
