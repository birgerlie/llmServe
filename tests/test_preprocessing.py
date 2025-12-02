"""Tests for text preprocessing."""

import pytest

from app.services.embedding_service import TextPreprocessor


class TestTextPreprocessor:
    """Tests for TextPreprocessor class."""

    def test_basic_preprocessing(self):
        """Test basic text preprocessing."""
        text = "  Dette er en test.  "
        result = TextPreprocessor.preprocess(text)
        assert result == "Dette er en test."

    def test_empty_input(self):
        """Test empty input handling."""
        assert TextPreprocessor.preprocess("") == ""
        assert TextPreprocessor.preprocess("   ") == ""

    def test_remove_courtesy_phrases(self):
        """Test removal of courtesy phrases."""
        text = "Vennligst send svar. Med vennlig hilsen, Ola"
        result = TextPreprocessor.preprocess(text)
        assert "vennligst" not in result.lower()
        assert "med vennlig hilsen" not in result.lower()

    def test_remove_mvh(self):
        """Test removal of 'mvh' abbreviation."""
        text = "Takk for hjelpen. Mvh. Test"
        result = TextPreprocessor.preprocess(text)
        assert "mvh" not in result.lower()

    def test_remove_urls(self):
        """Test removal of URLs."""
        text = "Besøk oss på www.example.no for mer info"
        result = TextPreprocessor.preprocess(text)
        assert "www.example.no" not in result

    def test_remove_https_urls(self):
        """Test removal of https URLs."""
        text = "Se https://example.com/page for detaljer"
        result = TextPreprocessor.preprocess(text)
        assert "https://" not in result

    def test_preserve_norwegian_chars(self):
        """Test that Norwegian characters are preserved."""
        text = "Æ, ø og å er norske bokstaver"
        result = TextPreprocessor.preprocess(text)
        assert "æ" in result.lower()
        assert "ø" in result.lower()
        assert "å" in result.lower()

    def test_normalize_whitespace(self):
        """Test whitespace normalization."""
        text = "Tekst   med    mye   mellomrom"
        result = TextPreprocessor.preprocess(text)
        assert "   " not in result
        assert "  " not in result

    def test_preserve_semantic_content(self):
        """Test that semantic content is preserved."""
        text = "Oslo kommune har ansvar for helsetjenester"
        result = TextPreprocessor.preprocess(text)
        assert "Oslo" in result
        assert "kommune" in result
        assert "helsetjenester" in result

    def test_no_noise_removal(self):
        """Test preprocessing without noise removal."""
        text = "Vennligst kontakt oss www.test.no"
        result = TextPreprocessor.preprocess(text, remove_noise=False)
        assert result == text

    def test_complex_text(self):
        """Test preprocessing of complex text."""
        text = """
        Sykehjemmet tilbyr:
        - Korttidsopphold
        - Langtidsopphold
        - Dagtilbud

        Kontakt: post@sykehjem.no
        Mvh. Administrasjonen
        """
        result = TextPreprocessor.preprocess(text)
        # Should preserve key information
        assert "korttidsopphold" in result.lower()
        assert "langtidsopphold" in result.lower()
        # Should remove contact info and courtesy phrases
        assert "mvh" not in result.lower()
