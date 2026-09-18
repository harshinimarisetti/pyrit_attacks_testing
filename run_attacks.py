# Multi-version import resolution for PyRIT converters
try:
    from pyrit.prompt_converter import Base64Converter, ROT13Converter, TranslationConverter
except ModuleNotFoundError:
    try:
        from pyrit.converter import Base64Converter, ROT13Converter, TranslationConverter
    except ModuleNotFoundError:
        from pyrit.prompt_normalizer import Base64Converter, ROT13Converter, TranslationConverter
