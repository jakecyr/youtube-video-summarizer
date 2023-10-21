from urllib.parse import ParseResult, parse_qsl, urlparse


class URLParser:

    """Parse a URL."""

    def __init__(self, url: str) -> None:
        """Initialize the URLParser instance.

        Args:
        ----
            url: The URL to parse.
        """
        self._result: ParseResult = urlparse(url)

    @property
    def scheme(self) -> str:
        """Returns the scheme of the URL.

        Returns
        -------
            A string representing the scheme of the URL.
        """
        return self._result.scheme

    @property
    def host(self) -> str:
        """Returns the host of the URL.

        Returns
        -------
            A string representing the host of the URL.
        """
        return self._result.netloc

    @property
    def path(self) -> str:
        """Returns the path of the URL.

        Returns
        -------
            A string representing the path of the URL.
        """
        return self._result.path

    @property
    def fragment(self) -> str:
        """Returns the fragment of the URL.

        Returns
        -------
            A string representing the fragment of the URL.
        """
        return self._result.fragment

    @property
    def query_string(self) -> str:
        """Returns the query string of the URL.

        Returns
        -------
            A string representing the query string of the URL.
        """
        return self._result.query

    def query_string_dict(self) -> dict:
        """Return the query string of the URL as a dictionary.

        Returns
        -------
            A dictionary representing the query string of the URL.
        """
        return dict(parse_qsl(self.query_string))
