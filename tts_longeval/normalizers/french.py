# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Our own tentative at making a French normalizer, in particular for numbers. Not heavily
tested and not as complete as the one for English developed by OpenAI. We use parser combinators
with `parsy`, which makes it quite nice and compact to declare new substitution rules."""
import re
from pathlib import Path

from parsy import regex, seq, success, whitespace, ParseError

from .basic import remove_symbols_and_diacritics


def word(reg: str):
    """Parser parsing a regex with a word boundary at the end (boundary not consumed)."""
    return regex(f'(?:{reg})' + r'\b')


def get_number_parser():
    """Inspired by `EnglishNumberNormalizer` from OpenAI, but implementing it using
    parser combinators.
    Not clear whether we need to be really thorough, but trying our best!
    It could be nice to extend it to also support physical units etc .
    We could also extend it to support ordinals, but there are so many specific rules!
    """
    mapping_path = Path(__file__).parent / "french_numbers.tsv"
    mapping = {}
    for line in mapping_path.read_text().split('\n'):
        line = line.strip()
        if not line:
            continue
        num, txt = line.strip().split('\t')
        mapping[txt] = int(num)

    keys = list(mapping.keys())
    # Sorting by decreasing order to ensure we maximally match.
    keys.sort(key=lambda x: -len(x))
    unites_et_dizaines = word(f"({'|'.join(keys)})").map(lambda x: mapping[x])
    ws = whitespace.result(" ")

    def multiplier(parser, previous_parser, allow_alone=True):
        # Lift up a parser for numbers below a certain threshold, to beyond that threshold and until the next.
        # E.g. parser should be something like `thousand`, `previous_parser` should be something that can
        # parse any number up to `999`, and allow_alone indicates whether it is correct to not put anything
        # in front of `thousand` (e.g. implicit 1 thousand).
        total = seq(previous_parser, (ws >> parser | success(1))).map(lambda a: a[0] * a[1])
        if allow_alone:
            total = parser | total
        return seq(total, (regex(r"(\s+et)?\s+") >> previous_parser) | success(0)).map(lambda a: a[0] + a[1])

    cent = word(r"cents?").result(100)
    up_to_cent = multiplier(cent, unites_et_dizaines)
    mille = word(r"mill(e|ier)s?").result(1000)
    up_to_mille = multiplier(mille, up_to_cent)
    millions = word(r"millions?").result(1_000_000)
    up_to_million = multiplier(millions, up_to_mille)
    milliard = word(r"milliards?").result(1_000_000_000)
    up_to_milliard = multiplier(milliard, up_to_million)

    number = up_to_milliard.map(lambda x: 'un' if x == 1 else str(x))

    euro = word("euros?").result("€")
    dollar = word("dollars?").result("$")
    pound = word(r"livres?(\s+sterling)?").result("£")
    centimes = word("centimes?")
    centimes_part = ((ws >> unites_et_dizaines) << (ws + centimes).optional()) | success(0)

    def _map_number_money(x):
        if x[1] is None:
            return x[0]
        else:
            if x[1][1] == 0:
                return x[0] + x[1][0]
            else:
                return f'{x[0]},{x[1][1]:02d}{x[1][0]}'

    number_or_money = seq(
        number,
        seq(ws + (euro | dollar | pound), centimes_part).optional()).map(_map_number_money)
    joker = regex(r'\w+')
    sep = regex(r'\W+')
    chunk = number_or_money | joker
    body = sep + chunk
    doc = (body | chunk) + body.many().concat()
    parser = doc.optional("") + sep.optional("")
    return parser


class FrenchNormalizer:
    """
    Removes accent, tries to normalize numbers and amounts (but not ordinals for now).
    """

    def __init__(self):
        self.ignore_patterns = r"\b(hmm|mm|mhm|mmm|uh|um)\b"
        self.parser = get_number_parser()

    def __call__(self, s: str) -> str:
        s = s.lower()
        s = re.sub(r"[<\[][^>\]]*[>\]]", "", s)  # remove words between brackets
        s = re.sub(r"\(([^)]+?)\)", "", s)  # remove words between parenthesis
        s = re.sub(self.ignore_patterns, "", s)
        s = re.sub(r"\,([^0-9]|$)", r" \1", s)  # remove periods not followed by numbers
        try:
            s = self.parser.parse(s)
        except ParseError as exc:
            index = exc.index
            start = max(0, index - 10)
            print("Original text:", s, s[start: start + 100])
            raise
        s = remove_symbols_and_diacritics(s, keep=",")  # keep numeric symbols
        s = re.sub(r"\s+", " ", s)
        return s


def test():
    parser = get_number_parser()
    print(parser.parse("cent mille"))
    print(parser.parse("cent vingt et un"))
    print(parser.parse("deux mille trois cents quarante euros vingt-trois"))
    print(parser.parse("deux mille trois cents quarante euros cinq centimes"))
    print(parser.parse("deux"))
    print(parser.parse("32 tamere cinqut"))
    print(parser.parse("deux milles milliards et trente-trois millions cent cinquante six"))
    print(parser.parse("puis dix-sans l'accusé"))


if __name__ == '__main__':
    test()
