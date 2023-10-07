from urllib.parse import ParseResult, parse_qsl, urlparse


class URLParser:
    def __init__(self, url: str) -> None:
        self._result: ParseResult = urlparse(url)

    @property
    def scheme(self) -> str:
        return self._result.scheme

    @property
    def host(self) -> str:
        return self._result.netloc

    @property
    def path(self) -> str:
        return self._result.path

    @property
    def fragment(self) -> str:
        return self._result.fragment

    @property
    def query_string(self) -> str:
        return self._result.query

    def query_string_dict(self) -> dict:
        return dict(parse_qsl(self.query_string))
