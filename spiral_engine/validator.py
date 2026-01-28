"""
validator.py - 유전 논리 검증기
가계도의 유전적 일관성을 검증
"""

from typing import List, Tuple, Dict
from .models import Family, Person, Gene
from .genetics import GeneticsEngine, InheritanceValidator


class LogicValidator:
    """가계도 논리 검증기"""

    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def validate(self, family: Family) -> Tuple[bool, List[str]]:
        """
        전체 가계도 검증

        Returns:
            (성공 여부, 오류 메시지 목록)
        """
        self.errors = []
        self.warnings = []

        # 1. 구조 검증
        self._validate_structure(family)

        # 2. 각 유전자에 대해 검증
        for gene in family.genes:
            self._validate_gene_inheritance(family, gene)

        return len(self.errors) == 0, self.errors

    def _validate_structure(self, family: Family):
        """가계도 구조 검증"""

        # 모든 구성원이 유전자형을 가지고 있는지 확인
        for person in family.members.values():
            for gene in family.genes:
                trait = person.get_trait(gene.symbol)
                if not trait:
                    self.errors.append(
                        f"{person.display_name}: {gene.symbol} 유전자형 없음"
                    )

        # 부모-자녀 관계 검증
        for person in family.members.values():
            if person.father_id:
                father = family.get_member(person.father_id)
                if not father:
                    self.errors.append(
                        f"{person.display_name}: 아버지 {person.father_id} 없음"
                    )

            if person.mother_id:
                mother = family.get_member(person.mother_id)
                if not mother:
                    self.errors.append(
                        f"{person.display_name}: 어머니 {person.mother_id} 없음"
                    )

    def _validate_gene_inheritance(self, family: Family, gene: Gene):
        """특정 유전자의 멘델 유전 법칙 검증"""

        for person in family.members.values():
            # 부모가 있는 경우에만 검증
            if not person.father_id or not person.mother_id:
                continue

            father = family.get_member(person.father_id)
            mother = family.get_member(person.mother_id)

            if not father or not mother:
                continue

            father_trait = father.get_trait(gene.symbol)
            mother_trait = mother.get_trait(gene.symbol)
            child_trait = person.get_trait(gene.symbol)

            if not all([father_trait, mother_trait, child_trait]):
                continue

            # 유전 가능 여부 검증
            is_valid = GeneticsEngine.is_valid_inheritance(
                gene,
                father_trait.genotype,
                mother_trait.genotype,
                child_trait.genotype,
                person.gender
            )

            if not is_valid:
                self.errors.append(
                    f"{gene.trait_name} 유전 오류: {person.display_name}의 "
                    f"유전자형 {child_trait.genotype}은 "
                    f"부모({father_trait.genotype} × {mother_trait.genotype})에서 불가능"
                )

    def get_validation_report(self, family: Family) -> Dict:
        """상세 검증 보고서"""
        is_valid, errors = self.validate(family)

        return {
            "valid": is_valid,
            "errors": errors,
            "warnings": self.warnings,
            "summary": {
                "total_members": len(family.members),
                "total_genes": len(family.genes),
                "error_count": len(errors),
                "warning_count": len(self.warnings)
            }
        }


def validate_logic(family: Family) -> Tuple[bool, List[str]]:
    """간편 검증 함수"""
    validator = LogicValidator()
    return validator.validate(family)
