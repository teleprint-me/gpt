"""
Module: gen.unicode

Copyright (c) 2023-2024 The ggml authors

References:
- Unicode Core Specification
    - https://www.unicode.org/versions/Unicode15.0.0/
- Unicode Data File Format
    - https://www.unicode.org/L2/L1999/UnicodeData.html
- Unicode Data Files
    - https://www.unicode.org/Public/UCD/latest/ucd/UnicodeData.txt
    - https://unicode.org/Public/UNIDATA/SpecialCasing.txt
    - https://www.unicode.org/Public/UCD/latest/ucd/LineBreak.txt
    - https://www.unicode.org/Public/UCD/latest/ucd/PropList.txt
- Unicode Algorithm
    - https://www.unicode.org/reports/tr9/
- Unicode Character Database
    - https://www.unicode.org/reports/tr44/
- Properties accessible through \\p{} and \\P{}
    - https://perldoc.perl.org/perluniprops
"""

import argparse
import array
import ctypes
import dataclasses
import logging
import unicodedata
from logging import Logger
from typing import Generator, Optional, Union

import requests

logger = logging.getLogger(__file__)


@dataclasses.dataclass(frozen=True)
class CodepointField:
    """
    Defines constants representing each field in a single line of the 'UnicodeData.txt'
    file with their corresponding index numbers (0-indexed). These values can be used to
    access specific fields when parsing and manipulating codepoints.

    These constants are useful when working with the `Codepoint` class and handling
    Unicode data in general.
    """

    CODE = 0  # Code value in 4-digit hexadecimal format.
    NAME = 1  # Character name
    GENERAL_CATEGORY = 2  # General Category
    CONONICAL_CC = 3  # Cononical Combining Classes
    BIDIRECTIONAL_CATEGORY = 4  # Bidirectional category
    DECOMPOSITION = 5  # Character decomposition mapping
    DECIMAL_DIGIT = 6  # Decimal digit value
    DIGIT = 7  # Digit value
    NUMERIC = 8  # Numeric value
    MIRRORED = 9  # Mirrored character flag
    OLD_NAME = 10  # Old Unicode name
    COMMENT = 11  # Comment field
    UPPERCASE = 12  # Uppercase mapping
    LOWERCASE = 13  # Lowercase mapping
    TITLECASE = 14  # Titlecase mapping


@dataclasses.dataclass
class Codepoint:
    """
    Represents a extracted codepoint from the 'UnicodeData.txt' file, containing
    information about various properties of a specific character or control code point.

    Field order is significant and corresponds to indices in the original data file.
    """

    code: int  # 0 Code value in 4-digit hexadecimal format.
    name: str  # 1 Character name
    general_category: str  # 2 General Category
    cononical_cc: int  # 3 Cononical Combining Classes
    bidirectional_category: str  # 4 Bidirectional category
    decomposition: tuple[str, int, ...]  # 5 Character decomposition mapping
    decimal_digit: int  # 6 Decimal digit value
    digit: int  # 7 Digit value
    numeric: str  # 8 Numeric value
    mirrored: bool  # 9 Mirrored character flag
    old_name: str  # 10 Old Unicode name
    comment: str  # 11 Comment field
    uppercase: int  # 12 Uppercase mapping
    lowercase: int  # 13 Lowercase mapping
    titlecase: int  # 14 Titlecase mapping

    def is_pair(self, other: "Codepoint") -> bool:
        """
        Check if a given Codepoint object (`other`) has the same mappings and
        properties compared to this current Codepoint instance (`self`).

        This method is used in `UnicodeDataRequest` class's `generate_codepoints()`
        generator function to ensure that pairs of characters (First, Last) are properly
        generated based on the 'UnicodeData.txt' file format.

        Args:
            other (Codepoint): A Codepoint instance representing another character

        Returns:
            bool: True if both instances have matching `lowercase`, `uppercase`,
            `general_category`, and `bidirectional_category` properties; otherwise False.
        """

        return (0, 0, self.general_category, self.bidirectional_category) == (
            other.lowercase,
            other.uppercase,
            other.general_category,
            other.bidirectional_category,
        )

    @staticmethod
    def parse_int(field: Union[int, str]) -> int:
        return int(field if field else "0", base=16)

    @staticmethod
    def parse_bool(field: str) -> bool:
        return True if field == "Y" else False

    @staticmethod
    def parse_decomposition(field: str) -> tuple[str, int, ...]:
        tokens = field.split(" ")
        special = [tokens[0]]
        values = [int(v, base=16) for v in tokens[1:]]
        return tuple(special + values)

    @classmethod
    def from_fields(cls, fields: list[str]) -> "Codepoint":
        """Returns a `Codepoint` instance object from a set of fields"""
        return Codepoint(
            code=int(fields[CodepointField.CODE], base=16),
            name=fields[CodepointField.NAME],
            general_category=fields[CodepointField.GENERAL_CATEGORY],
            cononical_cc=Codepoint.parse_int(fields[CodepointField.CONONICAL_CC]),
            bidirectional_category=fields[CodepointField.BIDIRECTIONAL_CATEGORY],
            decomposition=Codepoint.parse_decomposition(fields[CodepointField.DECOMPOSITION]),
            decimal_digit=fields[CodepointField.DECIMAL_DIGIT],
            digit=fields[CodepointField.DIGIT],
            numeric=fields[CodepointField.NUMERIC],
            mirrored=Codepoint.parse_bool(fields[CodepointField.MIRRORED]),
            old_name=fields[CodepointField.OLD_NAME],
            comment=fields[CodepointField.COMMENT],
            uppercase=Codepoint.parse_int(fields[CodepointField.UPPERCASE]),
            lowercase=Codepoint.parse_int(fields[CodepointField.LOWERCASE]),
            titlecase=Codepoint.parse_int(fields[CodepointField.TITLECASE]),
        )

    @classmethod
    def from_codepoint(cls, code: int, codepoint: "Codepoint") -> "Codepoint":
        """Returns a `Codepoint` instance object based on the range between First and Last"""
        return Codepoint(
            code=code,
            name=codepoint.name,
            general_category=codepoint.general_category,
            cononical_cc=codepoint.cononical_cc,
            bidirectional_category=codepoint.bidirectional_category,
            decomposition=codepoint.decomposition,
            decimal_digit=codepoint.decimal_digit,
            digit=codepoint.digit,
            numeric=codepoint.numeric,
            mirrored=codepoint.mirrored,
            old_name=codepoint.old_name,
            comment=codepoint.comment,
            uppercase=codepoint.uppercase,
            lowercase=codepoint.lowercase,
            titlecase=codepoint.titlecase,
        )


class UnicodeDataRequest:
    """
    A request object that fetches and processes Unicode data from an online repository.

    The requested data is processed as a generator, yielding instances of the Codepoint
    class representing individual characters or control codes in the Unicode Data Files format.

    Attributes:
        MAX_CODEPOINTS (int): Maximum number of code points to be fetched and generated
        UNICODE_DATA_URL (str): URL for fetching Unicode data from the online repository
        logger (Logger or None): A logger instance used for debugging purposes

    Methods:
        lines(self) -> list[str]: Returns a list of fetched unicode code points
        generate_codepoints(self) -> Generator[object, object, Codepoint]:
            Returns a generator to render code points dynamically
    """

    MAX_CODEPOINTS = 0x110000
    UNICODE_DATA_URL = "https://www.unicode.org/Public/UCD/latest/ucd/UnicodeData.txt"

    def __init__(
        self,
        url: Optional[str] = None,
        max_codepoints: Optional[int] = None,
        logger: Optional[Logger] = None,
    ):
        if max_codepoints is not None:
            self.MAX_CODEPOINTS = max_codepoints

        if url is not None:
            self.UNICODE_DATA_URL = url

        if logger is not None:
            self.logger = logger
        else:
            self.logger = Logger(self.__class__.__name__, level=logging.DEBUG)

    def lines(self) -> list[str]:
        """return the fetched unicode code points"""
        response = requests.get(self.UNICODE_DATA_URL)
        response.raise_for_status()
        data = response.content.decode()
        return data.splitlines()

    def generate_codepoints(self) -> Generator[object, object, Codepoint]:
        """return a generator to render codepoints dynamically"""
        previous = None
        for line in self.lines():
            # parse fields
            fields = line.split(";")
            message = f"line({line}): len({len(fields)}): fields({fields})"
            assert 15 == len(fields), message
            codepoint = Codepoint.from_fields(fields)
            # parse first
            if codepoint.name.endswith(", First>"):
                previous = codepoint
                continue
            # parse last
            if previous and codepoint.name.endswith(", Last>"):
                # yield only if codepoint subsets are valid
                message = f"Expected Last({codepoint}) after receiving First({previous})"
                assert codepoint.is_pair(previous), message
                for cpt in range(previous.code, codepoint.code):
                    yield Codepoint.from_codepoint(cpt, codepoint)
                previous = None
                continue
            yield codepoint


class CODEPOINT_FLAG:
    """
    Class representing Unicode codepoint properties as defined in the Unicode Standard Annex #9 (https://www.unicode.org/reports/tr9/)

    Attributes:
        UNDEFINED (int): Flag value for invalid or undefined codepoints
            (0x0001)
        NUMBER (int): Flag value for number properties
            (0x0002)
        LETTER (int): Flag value for letter properties
            (0x0004)
        SEPARATOR (int): Flag value for separator properties
            (0x0008)
        MARK (int): Flag value for mark properties
            (0x0010)
        PUNCTUATION (int): Flag value for punctuation properties
            (0x0020)
        SYMBOL (int): Flag value for symbol properties
            (0x0040)
        CONTROL (int): Flag value for control properties
            (0x0080)

    NOTE: See definition in unicode.h for implementation.
    """

    UNDEFINED = 0x0001  # invalid
    NUMBER = 0x0002  # \p{N}
    LETTER = 0x0004  # \p{L}
    SEPARATOR = 0x0008  # \p{Z}
    MARK = 0x0010  # \p{M}
    PUNCTUATION = 0x0020  # \p{P}
    SYMBOL = 0x0040  # \p{S}
    CONTROL = 0x0080  # \p{C}


class CODEPOINT_CATEGORY:
    """
    Class representing General Category properties as defined in Unicode Standard Annex #9
    (https://www.unicode.org/reports/tr9/) and other related resources.

    Attributes:
        FLAG (dict): Mapping of General Category names to corresponding CODEPOINT_FLAG values
            (e.g., {'Lu': CODEPOINT_FLAG.LETTER, ...})

    Notes:
        This class is based on the Normative and Informative Categories defined in
        Unicode Data Files (https://www.unicode.org/Public/UCD/)

    NOTE: General Category: https://www.unicode.org/L2/L1999/UnicodeData.html
    """

    FLAG = {
        # Normative Categories
        "Lu": CODEPOINT_FLAG.LETTER,  # Uppercase Letter
        "Ll": CODEPOINT_FLAG.LETTER,  # Lowercase Letter
        "Lt": CODEPOINT_FLAG.LETTER,  # Titlecase Letter
        "Mn": CODEPOINT_FLAG.MARK,  # Nonspacing Mark
        "Mc": CODEPOINT_FLAG.MARK,  # Spacing Mark
        "Me": CODEPOINT_FLAG.MARK,  # Enclosing Mark
        "Nd": CODEPOINT_FLAG.NUMBER,  # Decimal Number
        "Nl": CODEPOINT_FLAG.NUMBER,  # Letter Number
        "No": CODEPOINT_FLAG.NUMBER,  # Other Number
        "Zs": CODEPOINT_FLAG.SEPARATOR,  # Space Separator
        "Zl": CODEPOINT_FLAG.SEPARATOR,  # Line Separator
        "Zp": CODEPOINT_FLAG.SEPARATOR,  # Paragraph Separator
        "Cc": CODEPOINT_FLAG.CONTROL,  # Control
        "Cf": CODEPOINT_FLAG.CONTROL,  # Format
        "Cs": CODEPOINT_FLAG.CONTROL,  # Surrrogate
        "Co": CODEPOINT_FLAG.CONTROL,  # Private Use
        "Cn": CODEPOINT_FLAG.UNDEFINED,  # Undefined
        # Informative Categories
        "Lm": CODEPOINT_FLAG.LETTER,  # Modifier Letter
        "Lo": CODEPOINT_FLAG.LETTER,  # Other Letter
        "Pc": CODEPOINT_FLAG.PUNCTUATION,  # Connector Punctuation
        "Pd": CODEPOINT_FLAG.PUNCTUATION,  # Dash Punctuation
        "Ps": CODEPOINT_FLAG.PUNCTUATION,  # Open Punctuation
        "Pe": CODEPOINT_FLAG.PUNCTUATION,  # Close Punctuation
        "Pi": CODEPOINT_FLAG.PUNCTUATION,  # Initial Punctuation
        "Pf": CODEPOINT_FLAG.PUNCTUATION,  # Final Punctuation
        "Po": CODEPOINT_FLAG.PUNCTUATION,  # Other Punctuation
        "Sm": CODEPOINT_FLAG.SYMBOL,  # Math Symbol
        "Sc": CODEPOINT_FLAG.SYMBOL,  # Currency Symbol
        "Sk": CODEPOINT_FLAG.SYMBOL,  # Modifier Symbol
        "So": CODEPOINT_FLAG.SYMBOL,  # Other Symbol
    }


@dataclasses.dataclass
class UnicodeTable:
    """
    The `UnicodeTable` class serves as a container for various precomputed data related to Unicode characters, such as whitespace codes, lowercase and uppercase character ranges,
        normalized form D (NFD) mappings, etc., which can be used to improve the performance of text processing tasks.

    This class is primarily useful when working with large amounts of text data that require frequent lookups or manipulations based on Unicode properties,
        as it provides constant-time access to precomputed data instead of having to perform expensive computations at runtime.

    The `UnicodeTable` class can be initialized with empty lists for each property (whitespace, lowercase, uppercase, and nfd),
        but the recommended way is to load the necessary data from external files or databases during initialization to ensure accurate and up-to-date information.

    Here's an example of how you can create a `UnicodeTable` instance:

        ```python
        from tok.gguf import unicode

        table = unicode.UnicodeTable()

        # Load data for each property
        with open("whitespace_codes.txt", "r") as f:
            whitespaces = [int(line) for line in f]
            table.whitespace = whitespaces

        # ... continue loading other properties from external files or databases

        ```

    Once the `UnicodeTable` instance is initialized, you can access its properties using standard Python attribute syntax:

        ```python
        if 9 == table.whitespace[0]:
            print("The first whitespace code is a tab.")

        lowercase_range = (table.lowercase[0][0], table.lowercase[-1][1])
        print(f"Lowercase range: {ord('a')} - {lowercase_range[1]}")

        # ...

        ```
    """

    whitespace: list[int] = dataclasses.field(default_factory=list)
    lowercase: list[tuple[int, int]] = dataclasses.field(default_factory=list)
    uppercase: list[tuple[int, int]] = dataclasses.field(default_factory=list)
    nfd: list[tuple[int, int]] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class CodepointRanges:
    """
    The `CodepointRanges` class serves as a container for precomputed character ranges based on specific Unicode properties, such as character flags and normalized form D (NFD) mappings.

    This class is useful when working with large amounts of text data that require frequent lookups or manipulations based on Unicode properties,
        as it provides constant-time access to precomputed ranges instead of having to perform expensive computations at runtime.

    The `CodepointRanges` can be initialized with empty lists for each property (flags and nfd),
        but the recommended way is to load the necessary data from external files or databases during initialization to ensure accurate and up-to-date information.

    Here's an example of how you can create a `CodepointRanges` instance:

        ```python
        from tok.gguf import unicode

        ranges = unicode.CodepointRanges()

        # Load data for each property
        with open("flags_ranges.txt", "r") as f:
            flagged_codes = [tuple(map(int, line.split("-"))) for line in f]
            ranges.flags = flagged_codes

        # ... continue loading other properties from external files or databases

        ```

    Once the `CodepointRanges` instance is initialized, you can access its properties using standard Python attribute syntax:

        ```python
        for range in ranges.flags:
            start, end = range
            print(f"Flagged character range {start} - {end}")

        # ...

        ```
    """

    flags: list[tuple[int, int]] = dataclasses.field(default_factory=list)
    nfd: list[tuple[int, int, int]] = dataclasses.field(default_factory=list)


class CodepointProcessor:
    """
    The `CodepointProcessor` class precomputes various data related to Unicode characters, such as flags, whitespace codes, lowercase and uppercase character ranges, normalized form D (NFD) mappings, etc.,
        which can be used to improve the performance of text processing tasks. This class is primarily useful when working with large amounts of text data that require frequent lookups or manipulations based on Unicode properties,
            as it provides constant-time access to precomputed data instead of having to perform expensive computations at runtime.

    The `CodepointProcessor` can be initialized by specifying the maximum number of code points (Unicode characters) to process. If no limit is provided, all valid Unicode characters will be processed up to U+10FFFF.

    Once instantiated, you should call the `process_unicode()` method to compute and store precomputed data for each character within its defined limits. After processing,
        you can access various properties such as flags, whitespace codes, lowercase/uppercase ranges, normalized form D mappings, etc., using standard Python attribute syntax:

        ```python
        from tok.gguf import unicode

        processor = unicode.CodepointProcessor()
        processor.process_unicode()

        if 9 == processor.unicode_table.whitespace[0]:
            print("The first whitespace code is a tab.")

        lowercase_range = (processor.unicode_table.lowercase[0][0],
                           processor.unicode_table.lowercase[-1][1])

        uppercase_range = (processor.unicode_table.uppercase[0][0],
                           processor.unicode_table.uppercase[-1][1])

        print(f"Lowercase range: {ord('a')} - {lowercase_range[1]}")

        print(f"Uppercase range: {ord('A')} - {uppercase_range[1]}")

        # ...
        ```
    """

    def __init__(
        self,
        url: Optional[str] = None,
        max_codepoints: Optional[int] = None,
        logger: Optional[Logger] = None,
    ):
        # Set the unicode upper limit
        self._request = UnicodeDataRequest(url, max_codepoints, logger)

        if logger is not None:
            self.logger = logger
        else:
            self.logger = Logger(self.__class__.__name__, level=logging.DEBUG)

        # Set the unicode components
        initializer = [CODEPOINT_FLAG.UNDEFINED] * self.MAX_CODEPOINTS
        self._codepoint_flags = array.array("H", initializer)
        self._codepoint_ranges = CodepointRanges()
        self._unicode_table = UnicodeTable()

    @property
    def MAX_CODEPOINTS(self) -> int:
        return self._request.MAX_CODEPOINTS

    @property
    def codepoint_flags(self) -> array.ArrayType:
        return self._codepoint_flags

    @property
    def codepoint_ranges(self) -> CodepointRanges:
        return self._codepoint_ranges

    @property
    def unicode_table(self) -> UnicodeTable:
        return self._unicode_table

    def process_unicode(self):
        for codepoint in self._request.generate_codepoints():
            # convert codepoint to unicode character
            char = chr(codepoint.code)

            self.set_codepoint_flag(codepoint)
            self.set_lowercase_table(codepoint, char)
            self.set_uppercase_table(codepoint, char)
            self.set_nfd_table(codepoint, char)

        self.set_whitespace_table(codepoint, char)

    def set_codepoint_flag(self, codepoint: Codepoint) -> None:
        # codepoint category flag
        flag = CODEPOINT_CATEGORY.FLAG[codepoint.general_category]
        self._codepoint_flags[codepoint.code] = flag

    def set_lowercase_table(self, codepoint: Codepoint, char: str) -> None:
        if codepoint.lowercase:
            self._unicode_table.lowercase.append((codepoint.code, char))

    def set_uppercase_table(self, codepoint: Codepoint, char: str) -> None:
        if codepoint.uppercase:
            self._unicode_table.uppercase.append((codepoint.code, char))

    def set_nfd_table(self, codepoint: Codepoint, char: str):
        # NFD normalization
        norm = ord(unicodedata.normalize("NFD", char)[0])
        if codepoint != norm:
            self._unicode_table.nfd.append((codepoint.code, norm))

    def set_whitespace_table(self, codepoint: Codepoint, char: str):
        # whitespaces, see "<White_Space>" https://www.unicode.org/Public/UCD/latest/ucd/PropList.txt
        self._unicode_table.whitespace.extend(range(0x0009, 0x000D + 1))
        self._unicode_table.whitespace.extend(range(0x2000, 0x200A + 1))
        self._unicode_table.whitespace.extend(
            [0x0020, 0x0085, 0x00A0, 0x1680, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000]
        )

    def group_flag_ranges(self):
        # group ranges with same flags
        self._codepoint_ranges.flags = [(0, self._codepoint_flags[0])]  # start, flags
        for codepoint, flag in enumerate(self._codepoint_flags):
            if flag != self._codepoint_ranges.flags[-1][1]:
                self._codepoint_ranges.flags.append((codepoint, flag))
        self._codepoint_ranges.flags.append((self.MAX_CODEPOINTS, 0x0000))

    def group_nfd_ranges(self):
        # group ranges with same nfd
        self._codepoint_ranges.nfd = [(0, 0, 0)]  # start, last, nfd
        for codepoint, norm in self._unicode_table.nfd:
            start = self._codepoint_ranges.nfd[-1][0]
            if self._codepoint_ranges.nfd[-1] != (start, codepoint - 1, norm):
                self._codepoint_ranges.nfd.append(None)
                start = codepoint
            self._codepoint_ranges.nfd[-1] = (start, codepoint, norm)


"""
Module: gen.unicode

Generate 'unicode-data.cpp' and 'unicode-data.h'
"""


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate 'unicode-data.cpp' and 'unicode-data.h'")

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Output generated source text (default: False)",
    )

    # output path - default to stdout if no output path is given
    parser.add_argument(
        "--output-path",
        type=str,
        help="Output file path (default: stdout)",
    )

    # endianess - default to little-endian if no option provided
    parser.add_argument(
        "--big-endian",
        action="store_true",
        help="The byte order of the code points (default: False ('little'))",
    )

    # max_codepoints - default to 0x110000 if no boundary is given
    parser.add_argument(
        "--max-codepoints",
        type=int,
        default=0x110000,
        help="Maximum code points limit (default: 0x110000)",
    )

    return parser.parse_args()


def build_unicode_data_h(max_codepoints: int = 0x110000) -> str:
    # NOTE: The resulting string is segmented to prevent formatting conflicts with braces
    unicode_data_h = """\
    // generated with python gguf.cli.unicode
    #ifndef UNICODE_DATA_H
    #define UNICODE_DATA_H

    #include <cstdint>
    #include <vector>
    #include <unordered_map>
    #include <unordered_set>

    /**
     * @brief Represents a Unicode character range with normalized form D (NFD)
     */
    struct range_nfd {
        uint32_t first;
        uint32_t last;
        uint32_t nfd;
    };\n
    """

    unicode_data_h += f"""\
    static const uint32_t MAX_CODEPOINTS = {max_codepoints};\n
    """

    unicode_data_h += """\
    /**
     * @brief Externally linked variables for Unicode data structures
     */
    extern const std::vector<std::pair<uint32_t, uint16_t>> unicode_ranges_flags;
    extern const std::unordered_set<uint32_t> unicode_set_whitespace;
    extern const std::unordered_map<uint32_t, uint32_t> unicode_map_lowercase;
    extern const std::unordered_map<uint32_t, uint32_t> unicode_map_uppercase;
    extern const std::vector<range_nfd> unicode_ranges_nfd;
    #endif // UNICODE_DATA_H
    """

    # NOTE: Format source text by line
    return "\n".join([line.strip() for line in unicode_data_h.split("\n")])


# TODO: define helper functions for setting mapping?
def set_ranges_flags(processor: CodepointProcessor, byte_order: str = "little") -> str:
    unicode_ranges_flags = (
        "// codepoint, flag // last=next_start-1\n"
        "const std::vector<std::pair<uint32_t, uint16_t>> unicode_ranges_flags = {\n"
    )
    logger.debug(unicode_ranges_flags)

    for codepoint, flags in processor.codepoint_ranges.flags:
        flags = int.from_bytes(bytes(flags), byte_order)
        line = "{0x%06X, 0x%04X}," % (codepoint, flags)
        logger.debug(line)
        unicode_ranges_flags += line

    line = "};\n\n"
    logger.debug(line)

    return unicode_ranges_flags + line


def set_unicode_whitespace(processor: CodepointProcessor) -> str:
    unicode_set_whitespace = "const std::unordered_set<uint32_t> unicode_set_whitespace = {\n"
    logger.debug(unicode_set_whitespace)

    for codepoint in processor.unicode_table.whitespace:
        line = "0x%06X" % codepoint
        logger.debug(line)
        unicode_set_whitespace += f"{line}, "

    line = "};\n\n"
    logger.debug(line)

    return unicode_set_whitespace + line


def set_unicode_lowercase(processor: CodepointProcessor) -> str:
    unicode_map_lowercase = (
        "const std::unordered_map<uint32_t, uint32_t> unicode_map_lowercase = {\n"
    )

    for tuple in processor.unicode_table.lowercase:
        line = "{0x%06X, 0x%06X}," % tuple
        logger.debug(line)
        unicode_map_lowercase += line

    line = "};\n\n"
    logger.debug(line)

    return unicode_map_lowercase + line


def set_unicode_uppercase(processor: CodepointProcessor) -> str:
    unicode_map_uppercase = (
        "const std::unordered_map<uint32_t, uint32_t> unicode_map_uppercase = {\n"
    )

    for tuple in processor.unicode_table.uppercase:
        line = "{0x%06X, 0x%06X}," % tuple
        logger.debug(line)
        unicode_map_uppercase += line

    line = "};\n\n"
    logger.debug(line)

    return unicode_map_uppercase + line


def set_ranges_nfd(processor: CodepointProcessor) -> str:
    unicode_ranges_nfd = (
        "// start, last, nfd\n" "const std::vector<range_nfd> unicode_ranges_nfd = {\n"
    )

    for triple in processor.codepoint_ranges.nfd:
        line = "{0x%06X, 0x%06X, 0x%06X}," % triple
        logger.debug(line)
        unicode_ranges_nfd += line

    line = "};\n"
    logger.debug(line)

    return unicode_ranges_nfd + line


def build_unicode_data_cpp(processor: CodepointProcessor) -> str:
    # define includes
    unicode_data_cpp = """
    // generated with python gguf.cli.unicode

    #include "unicode-data.h"

    #include <cstdint>
    #include <vector>
    #include <unordered_map>
    #include <unordered_set>\n
    """
    logger.debug(unicode_data_cpp)

    unicode_data_cpp += set_ranges_flags(processor)
    unicode_data_cpp += set_unicode_whitespace(processor)
    unicode_data_cpp += set_unicode_lowercase(processor)
    unicode_data_cpp += set_unicode_uppercase(processor)
    unicode_data_cpp += set_ranges_nfd(processor)

    return unicode_data_cpp


def main():
    args = get_arguments()

    if args.verbose or not args.output_path:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    processor = CodepointProcessor(args.max_codepoints)
    processor.process_unicode()
    processor.group_flag_ranges()
    processor.group_nfd_ranges()

    # build the header file
    unicode_data_h = build_unicode_data_h(args.max_codepoints)

    # build the source file
    unicode_data_cpp = build_unicode_data_cpp(processor)

    if args.output_path:
        header_file = f"{args.output_path}/unicode-data.h"
        cpp_file = f"{args.output_path}/unicode-data.cpp"

        with open(header_file, "w") as f:
            f.write(unicode_data_h)

        with open(cpp_file, "w") as f:
            f.write(unicode_data_cpp)


if __name__ == "__main__":
    main()
