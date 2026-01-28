"""
Spiral Engine - 수능 생명과학 유전 문제 생성 엔진
Korean SAT Biology Genetics Problem Generator

이 엔진은 논리적 오류가 없는 유전 가계도 문제를 자동 생성합니다.
"""

from .models import Person, Gene, GeneticTrait, Family
from .genetics import InheritanceMode, InheritanceRule
from .generator import PedigreeGenerator
from .data_table import DNATableGenerator
from .visualizer import PedigreeVisualizer
from .validator import LogicValidator

__version__ = "1.0.0"
__author__ = "Spiral Engine Team"

__all__ = [
    'Person',
    'Gene',
    'GeneticTrait',
    'Family',
    'InheritanceMode',
    'InheritanceRule',
    'PedigreeGenerator',
    'DNATableGenerator',
    'PedigreeVisualizer',
    'LogicValidator',
]
