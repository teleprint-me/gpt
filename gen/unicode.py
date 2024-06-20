"""
Module: gen.unicode

Copyright (c) 2023-2024 The ggml authors

References:
- Unicode Chapter 3 Unicode Conformance
    - https://www.unicode.org/versions/Unicode15.0.0/ch03.pdf
- Properties accessible through \\p{} and \\P{}
    - https://perldoc.perl.org/perluniprops
- Ctypes
    - https://docs.python.org/3/library/ctypes.html
- Data Parsing
    - https://www.unicode.org/L2/L1999/UnicodeData.html
- Data Raw
    - https://www.unicode.org/Public/UCD/latest/ucd/UnicodeData.txt
- Unicode Algorithm
    - https://www.unicode.org/reports/tr9/
"""

import argparse
import ctypes
import dataclasses
import logging
import unicodedata

import regex

logger = logging.getLogger(__file__)


class CodepointFlags(ctypes.Structure):
    """
    Represents Unicode character properties as defined by the Unicode Technical Standard #36 (Unicode 5.2) using Python's ctypes library,
        providing boolean flags for various categories of characters based on their unicode values and properties.

    This class allows developers to easily check if a given code point belongs to specific character categories such as numbers (\\p{N}),
        letters (\\p{L}), separators (\\p{Z}), accent marks (\\p{M}), punctuation (\\p{P}), symbols (\\p{S}), and controls (\\p{C}).

    The `CodepointFlags` class uses a structure defined in the unicode.h header file to store these properties efficiently,
        making it suitable for high-performance applications that need to process large amounts of text data with Unicode support.

    To use this class, create an instance of CodepointFlags and call its `from_codepoints` method passing a list or iterable containing the code points
        you want to check, e.g.:

        ```python
        from tok.gguf import unicode

        flags = unicode.CodepointFlags()
        flagged_chars = [0x2145, 0x65, 0xFFFD]

        flags.from_codepoints(flagged_chars)

        for codepoint in flagged_chars:
            print(f"{codepoint}: is_number={flags.is_number(codepoint)}, is_letter={flags.is_letter(codepoint)}")
        ```
    """

    _fields_ = [  # see definition in unicode.h
        ("is_undefined", ctypes.c_uint16, 1),
        ("is_number", ctypes.c_uint16, 1),  # regex: \p{N}
        ("is_letter", ctypes.c_uint16, 1),  # regex: \p{L}
        ("is_separator", ctypes.c_uint16, 1),  # regex: \p{Z}
        ("is_accent_mark", ctypes.c_uint16, 1),  # regex: \p{M}
        ("is_punctuation", ctypes.c_uint16, 1),  # regex: \p{P}
        ("is_symbol", ctypes.c_uint16, 1),  # regex: \p{S}
        ("is_control", ctypes.c_uint16, 1),  # regex: \p{C}
    ]


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

    def __init__(self, max_codepoints: None | int = 0x110000):
        # Set the unicode upper limit
        self.MAX_CODEPOINTS = max_codepoints

        # Regular expressions for various Unicode character categories
        self._regexes = {
            "is_number": regex.compile(r"\p{N}"),
            "is_letter": regex.compile(r"\p{L}"),
            "is_separator": regex.compile(r"\p{Z}"),
            "is_accent_mark": regex.compile(r"\p{M}"),
            "is_punctuation": regex.compile(r"\p{P}"),
            "is_symbol": regex.compile(r"\p{S}"),
            "is_control": regex.compile(r"\p{C}"),
            "is_whitespace": regex.compile(r"\s"),
        }

        # Set the unicode components
        self._codepoint_flags = (CodepointFlags * self.MAX_CODEPOINTS)()
        self._codepoint_ranges = CodepointRanges()
        self._unicode_table = UnicodeTable()

    @property
    def codepoint_flags(self) -> CodepointFlags:
        return self._codepoint_flags

    @property
    def codepoint_ranges(self) -> CodepointRanges:
        return self._codepoint_ranges

    @property
    def unicode_table(self) -> UnicodeTable:
        return self._unicode_table

    def process_unicode(self):
        for codepoint in range(self.MAX_CODEPOINTS):
            # convert codepoint to unicode character
            char = chr(codepoint)

            # regex categories
            flags = self._codepoint_flags[codepoint]
            for flag in dir(flags):
                if flag.startswith("__"):
                    continue

                regex = self._regexes.get(flag)
                if regex is not None:
                    setattr(flags, flag, bool(regex.match(char)))
                elif flag == "is_undefined":
                    setattr(flags, flag, bytes(flags)[0] == 0)
                    message = f"'flags' is undefined for {codepoint} with {char}"
                    assert not flags.is_undefined, message

            self.set_whitespace_table(codepoint, char)
            self.set_lowercase_table(codepoint, char)
            self.set_uppercase_table(codepoint, char)
            self.set_nfd_table(codepoint, char)

    def set_whitespace_table(self, codepoint: int, char: str):
        # whitespace
        regex = self._regexes["is_whitespace"]
        if bool(regex.match(char)):
            self._unicode_table.whitespace.append(codepoint)

    def set_lowercase_table(self, codepoint: int, char: str):
        # lowercase conversion
        lower = ord(char.lower()[0])
        if codepoint != lower:
            self._unicode_table.lowercase.append((codepoint, lower))

    def set_uppercase_table(self, codepoint: int, char: str):
        # uppercase conversion
        upper = ord(char.upper()[0])
        if codepoint != upper:
            self._unicode_table.uppercase.append((codepoint, upper))

    def set_nfd_table(self, codepoint: int, char: str):
        # NFD normalization
        norm = ord(unicodedata.normalize("NFD", char)[0])
        if codepoint != norm:
            self._unicode_table.nfd.append((codepoint, norm))

    def group_flag_ranges(self):
        # group ranges with same flags
        self._codepoint_ranges.flags = [(0, self._codepoint_flags[0])]  # start, flags
        for codepoint, flags in enumerate(self._codepoint_flags):
            if bytes(flags) != bytes(self._codepoint_ranges.flags[-1][1]):
                self._codepoint_ranges.flags.append((codepoint, flags))
        self._codepoint_ranges.flags.append((self.MAX_CODEPOINTS, CodepointFlags()))

    def group_nfd_ranges(self):
        # group ranges with same nfd
        self._codepoint_ranges.nfd = [(0, 0, 0)]  # start, last, nfd
        for codepoint, norm in self.unicode_table.nfd:
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
    parser = argparse.ArgumentParser(
        description="Generate 'unicode-data.cpp' and 'unicode-data.h'"
    )

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
    unicode_set_whitespace = (
        "const std::unordered_set<uint32_t> unicode_set_whitespace = {\n"
    )
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
    assert ctypes.sizeof(CodepointFlags) == 2

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
