"""Helper functions for constructing LTL formulas."""

from .parser import (
    LTLFormula, LTLFormulaBinaryOp, LTLFormulaUnaryOp,
    LTLFormulaLeaf, Token
)


def OR(x: LTLFormula, y: LTLFormula) -> LTLFormula:
    """Create an OR formula."""
    return LTLFormulaBinaryOp(Token.OR, x, y)


def AND(x: LTLFormula, y: LTLFormula) -> LTLFormula:
    """Create an AND formula."""
    return LTLFormulaBinaryOp(Token.AND, x, y)


def IMPLIES(x: LTLFormula, y: LTLFormula) -> LTLFormula:
    """Create an IMPLIES formula."""
    return LTLFormulaBinaryOp(Token.IMPLIES, x, y)


def NEXT(x: LTLFormula) -> LTLFormula:
    """Create a NEXT formula."""
    return LTLFormulaUnaryOp(Token.NEXT, x)


def GLOBALLY(x: LTLFormula) -> LTLFormula:
    """Create a GLOBALLY (G) formula."""
    return LTLFormulaUnaryOp(Token.GLOBALLY, x)


def EVENTUALLY(x: LTLFormula) -> LTLFormula:
    """Create an EVENTUALLY (F) formula."""
    return LTLFormulaUnaryOp(Token.EVENTUALLY, x)


def NOT(x: LTLFormula) -> LTLFormula:
    """Create a NOT formula."""
    return LTLFormulaUnaryOp(Token.NOT, x)


def AP(s: str) -> LTLFormula:
    """Create an atomic proposition."""
    return LTLFormulaLeaf(Token.AP, s)
