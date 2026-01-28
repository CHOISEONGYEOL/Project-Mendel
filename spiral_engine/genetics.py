"""
genetics.py - 유전 규칙 및 상속 로직
InheritanceMode, InheritanceRule 클래스
"""

from enum import Enum
from typing import List, Tuple, Optional, Set, Dict
from itertools import product
import random

from .models import (
    Gene, GeneticTrait, Person, Family,
    Gender, Phenotype, ChromosomeType
)


class InheritanceMode(Enum):
    """유전 방식"""
    AUTOSOMAL_DOMINANT = "상염색체 우성"
    AUTOSOMAL_RECESSIVE = "상염색체 열성"
    X_LINKED_DOMINANT = "X염색체 연관 우성"
    X_LINKED_RECESSIVE = "X염색체 연관 열성"


class InheritanceRule:
    """
    유전 규칙 클래스 - 멘델 유전 법칙 구현
    """

    @staticmethod
    def get_possible_offspring_genotypes(
        father: Person,
        mother: Person,
        gene: Gene
    ) -> List[Tuple[str, Optional[str]]]:
        """
        부모의 유전자형으로부터 가능한 자녀 유전자형 조합 반환

        Returns:
            List of (allele1, allele2) tuples
        """
        father_trait = father.get_trait(gene.symbol)
        mother_trait = mother.get_trait(gene.symbol)

        if not father_trait or not mother_trait:
            return []

        father_gametes = father_trait.get_possible_gametes()
        mother_gametes = mother_trait.get_possible_gametes()

        if gene.chromosome_type == ChromosomeType.X_LINKED:
            # X-linked: 아버지는 딸에게만 X 전달, 아들에게는 Y
            # 어머니는 항상 X 전달
            possible = []

            # 딸의 경우 (아버지의 X + 어머니의 X 중 하나)
            for m_allele in mother_gametes:
                for f_allele in father_gametes:
                    possible.append((f_allele, m_allele))  # 여성용

            # 아들의 경우 (어머니의 X 중 하나만)
            for m_allele in mother_gametes:
                possible.append((m_allele, None))  # 남성용

            return possible
        else:
            # Autosomal: 양쪽 부모에서 하나씩
            possible = []
            for f_allele in father_gametes:
                for m_allele in mother_gametes:
                    possible.append((f_allele, m_allele))
            return possible

    @staticmethod
    def get_possible_offspring_for_gender(
        father: Person,
        mother: Person,
        gene: Gene,
        child_gender: Gender
    ) -> List[Tuple[str, Optional[str]]]:
        """
        특정 성별 자녀의 가능한 유전자형 반환
        """
        father_trait = father.get_trait(gene.symbol)
        mother_trait = mother.get_trait(gene.symbol)

        if not father_trait or not mother_trait:
            return []

        father_gametes = father_trait.get_possible_gametes()
        mother_gametes = mother_trait.get_possible_gametes()

        if gene.chromosome_type == ChromosomeType.X_LINKED:
            if child_gender == Gender.FEMALE:
                # 딸: 아버지 X + 어머니 X
                return [(f, m) for f in father_gametes for m in mother_gametes]
            else:
                # 아들: 어머니 X만 (아버지에서 Y 받음)
                return [(m, None) for m in mother_gametes]
        else:
            # Autosomal: 성별 무관
            return [(f, m) for f in father_gametes for m in mother_gametes]

    @staticmethod
    def can_produce_phenotype(
        father: Person,
        mother: Person,
        gene: Gene,
        target_phenotype: Phenotype,
        child_gender: Gender
    ) -> bool:
        """
        부모가 특정 표현형의 자녀를 낳을 수 있는지 확인
        """
        possible = InheritanceRule.get_possible_offspring_for_gender(
            father, mother, gene, child_gender
        )

        for allele1, allele2 in possible:
            trait = GeneticTrait(gene, allele1, allele2)
            if trait.phenotype == target_phenotype:
                return True
        return False

    @staticmethod
    def get_valid_offspring_genotypes(
        father: Person,
        mother: Person,
        gene: Gene,
        target_phenotype: Phenotype,
        child_gender: Gender
    ) -> List[Tuple[str, Optional[str]]]:
        """
        목표 표현형을 만족하는 자녀 유전자형 목록 반환
        """
        possible = InheritanceRule.get_possible_offspring_for_gender(
            father, mother, gene, child_gender
        )

        valid = []
        for allele1, allele2 in possible:
            trait = GeneticTrait(gene, allele1, allele2)
            if trait.phenotype == target_phenotype:
                valid.append((allele1, allele2))
        return valid

    @staticmethod
    def get_required_parent_genotypes(
        gene: Gene,
        child_genotype: Tuple[str, Optional[str]],
        child_gender: Gender,
        parent_gender: Gender
    ) -> List[Tuple[str, Optional[str]]]:
        """
        자녀의 유전자형을 만들 수 있는 부모의 가능한 유전자형 반환

        Args:
            gene: 유전자
            child_genotype: 자녀 유전자형 (allele1, allele2)
            child_gender: 자녀 성별
            parent_gender: 부모 성별

        Returns:
            가능한 부모 유전자형 목록
        """
        child_allele1, child_allele2 = child_genotype
        d_allele = gene.dominant_allele
        r_allele = gene.recessive_allele

        if gene.chromosome_type == ChromosomeType.X_LINKED:
            if parent_gender == Gender.MALE:
                # 아버지 (XY)
                if child_gender == Gender.FEMALE:
                    # 딸에게 전달한 대립유전자를 가져야 함
                    # 아버지가 딸에게 준 것은 child_allele1 (정렬 전 기준 추정 필요)
                    # X-linked에서 아버지는 X 하나만 가짐
                    # 딸이 받은 것 중 아버지에게서 온 것 확인
                    possible = []
                    if child_allele1:
                        possible.append((child_allele1, None))
                    if child_allele2 and child_allele2 != child_allele1:
                        possible.append((child_allele2, None))
                    return list(set(possible))
                else:
                    # 아들에게는 Y만 주므로 아버지 유전자형은 무관
                    return [(d_allele, None), (r_allele, None)]
            else:
                # 어머니 (XX)
                if child_gender == Gender.FEMALE:
                    # 딸에게 준 대립유전자 중 하나를 가져야 함
                    possible = []
                    for maternal_allele in [child_allele1, child_allele2]:
                        if maternal_allele:
                            # 어머니가 이 대립유전자를 가질 수 있는 유전자형
                            if maternal_allele.isupper():
                                possible.extend([
                                    (d_allele, d_allele),
                                    (d_allele, r_allele)
                                ])
                            else:
                                possible.extend([
                                    (d_allele, r_allele),
                                    (r_allele, r_allele)
                                ])
                    return list(set(possible))
                else:
                    # 아들 (어머니의 X만 받음)
                    son_allele = child_allele1
                    if son_allele.isupper():
                        return [(d_allele, d_allele), (d_allele, r_allele)]
                    else:
                        return [(d_allele, r_allele), (r_allele, r_allele)]
        else:
            # Autosomal
            # 자녀는 각 부모에게서 하나씩 받음
            possible = []
            for child_allele in [child_allele1, child_allele2]:
                if child_allele:
                    if child_allele.isupper():
                        possible.extend([
                            (d_allele, d_allele),
                            (d_allele, r_allele)
                        ])
                    else:
                        possible.extend([
                            (d_allele, r_allele),
                            (r_allele, r_allele)
                        ])
            return list(set(possible))

    @staticmethod
    def infer_parent_genotypes_from_children(
        gene: Gene,
        children: List[Person],
        parent_gender: Gender
    ) -> Set[Tuple[str, ...]]:
        """
        자녀들의 유전자형으로부터 부모의 가능한 유전자형 추론
        모든 자녀를 낳을 수 있는 유전자형만 반환
        """
        if not children:
            # 자녀가 없으면 모든 유전자형 가능
            d = gene.dominant_allele
            r = gene.recessive_allele
            if gene.chromosome_type == ChromosomeType.X_LINKED:
                if parent_gender == Gender.MALE:
                    return {(d,), (r,)}
                else:
                    return {(d, d), (d, r), (r, r)}
            else:
                return {(d, d), (d, r), (r, r)}

        # 각 자녀별로 가능한 부모 유전자형 계산
        possible_sets = []
        for child in children:
            trait = child.get_trait(gene.symbol)
            if trait:
                genotype = (trait.allele1, trait.allele2) if trait.allele2 else (trait.allele1,)
                possible = InheritanceRule.get_required_parent_genotypes(
                    gene, (trait.allele1, trait.allele2),
                    child.gender, parent_gender
                )
                possible_sets.append(set(possible))

        if not possible_sets:
            return set()

        # 교집합: 모든 자녀를 낳을 수 있는 유전자형
        result = possible_sets[0]
        for ps in possible_sets[1:]:
            result = result.intersection(ps)

        return result

    @staticmethod
    def validate_inheritance(
        father: Person,
        mother: Person,
        child: Person,
        gene: Gene
    ) -> Tuple[bool, str]:
        """
        부모-자녀 간 유전이 유효한지 검증

        Returns:
            (is_valid, error_message)
        """
        child_trait = child.get_trait(gene.symbol)
        if not child_trait:
            return True, ""  # 형질 정보 없으면 검증 스킵

        father_trait = father.get_trait(gene.symbol)
        mother_trait = mother.get_trait(gene.symbol)

        if not father_trait or not mother_trait:
            return False, f"부모의 {gene.symbol} 형질 정보 없음"

        # 가능한 자녀 유전자형 확인
        possible = InheritanceRule.get_possible_offspring_for_gender(
            father, mother, gene, child.gender
        )

        child_genotype = (child_trait.allele1, child_trait.allele2)

        # 순서 무관하게 비교 (상염색체의 경우)
        for allele1, allele2 in possible:
            if gene.chromosome_type == ChromosomeType.AUTOSOMAL:
                # 상염색체: 순서 무관
                child_set = {child_trait.allele1, child_trait.allele2}
                possible_set = {allele1, allele2}
                if child_set == possible_set:
                    return True, ""
            else:
                # X-linked: 남성은 하나만
                if child.gender == Gender.MALE:
                    if child_trait.allele1 == allele1 and child_trait.allele2 == allele2:
                        return True, ""
                else:
                    child_set = {child_trait.allele1, child_trait.allele2}
                    possible_set = {allele1, allele2}
                    if child_set == possible_set:
                        return True, ""

        return False, (
            f"자녀 {child.id}의 유전자형 {child_trait.genotype}은 "
            f"부모({father_trait.genotype} × {mother_trait.genotype})에서 나올 수 없음"
        )


class GenotypeConstraint:
    """유전자형 제약 조건"""

    def __init__(
        self,
        person_id: str,
        gene_symbol: str,
        phenotype: Optional[Phenotype] = None,
        must_be_carrier: Optional[bool] = None,
        specific_genotype: Optional[Tuple[str, Optional[str]]] = None
    ):
        self.person_id = person_id
        self.gene_symbol = gene_symbol
        self.phenotype = phenotype
        self.must_be_carrier = must_be_carrier
        self.specific_genotype = specific_genotype

    def is_satisfied(self, person: Person, gene: Gene) -> bool:
        """제약 조건 충족 여부"""
        trait = person.get_trait(gene.symbol)
        if not trait:
            return False

        if self.phenotype and trait.phenotype != self.phenotype:
            return False

        if self.must_be_carrier is not None:
            if trait.is_carrier != self.must_be_carrier:
                return False

        if self.specific_genotype:
            current = (trait.allele1, trait.allele2)
            if gene.chromosome_type == ChromosomeType.AUTOSOMAL:
                if set(current) != set(self.specific_genotype):
                    return False
            else:
                if current != self.specific_genotype:
                    return False

        return True


class ProblemScenario:
    """
    문제 시나리오 정의
    수능 문제 유형별 설정
    """

    @staticmethod
    def affected_child_from_normal_parents(gene: Gene) -> Dict:
        """정상 부모에서 유전병 자녀가 태어난 경우"""
        return {
            'description': '정상 부모 사이에서 유전병 자녀 발생',
            'parent_phenotype': Phenotype.DOMINANT,
            'child_phenotype': Phenotype.RECESSIVE,
            'inheritance_hint': '열성 유전',
            'gene': gene
        }

    @staticmethod
    def carrier_detection(gene: Gene) -> Dict:
        """보인자 확인 문제"""
        return {
            'description': '보인자 확인 문제',
            'requires_carrier': True,
            'gene': gene
        }

    @staticmethod
    def x_linked_pattern(gene: Gene) -> Dict:
        """X-linked 유전 패턴"""
        return {
            'description': 'X염색체 연관 유전 패턴',
            'chromosome_type': ChromosomeType.X_LINKED,
            'gene': gene
        }
