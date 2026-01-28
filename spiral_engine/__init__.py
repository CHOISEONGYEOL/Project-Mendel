"""
Spiral Engine - 수능 생명과학 유전 문제 생성기

수능 스타일의 가계도 문제를 자동으로 생성하는 엔진
"""

from .models import (
    Gender,
    Phenotype,
    InheritanceMode,
    GeneticTrait,
    Gene,
    Person,
    Family
)

from .genetics import (
    GeneticsEngine,
    InheritanceValidator
)

from .generator import (
    GenerationConfig,
    PedigreeGenerator
)

from .visualizer import (
    GridConfig,
    PedigreeVisualizer,
)

from .validator import (
    LogicValidator,
    validate_logic
)

from .data_table import (
    DNATableConfig,
    DNATableGenerator
)


__version__ = "1.0.0"
__all__ = [
    # Models
    "Gender",
    "Phenotype",
    "InheritanceMode",
    "GeneticTrait",
    "Gene",
    "Person",
    "Family",

    # Genetics
    "GeneticsEngine",
    "InheritanceValidator",

    # Generator
    "GenerationConfig",
    "PedigreeGenerator",

    # Visualizer
    "GridConfig",
    "PedigreeVisualizer",

    # Validator
    "LogicValidator",
    "validate_logic",

    # Data Table
    "DNATableConfig",
    "DNATableGenerator",
]
