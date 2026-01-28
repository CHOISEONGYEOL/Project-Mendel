"""
validator.py - 논리 검증 모듈
생성된 가계도 데이터의 유전학적 정합성 검증
"""

from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
from enum import Enum

from .models import Family, Person, Gene, GeneticTrait, Gender, Phenotype, ChromosomeType
from .genetics import InheritanceRule
from .data_table import DNATable


class ValidationLevel(Enum):
    """검증 레벨"""
    ERROR = "ERROR"      # 치명적 오류 (유전학 법칙 위반)
    WARNING = "WARNING"  # 경고 (드문 케이스)
    INFO = "INFO"        # 정보


@dataclass
class ValidationResult:
    """검증 결과"""
    is_valid: bool
    level: ValidationLevel
    message: str
    details: Dict = field(default_factory=dict)

    def __str__(self):
        return f"[{self.level.value}] {self.message}"


@dataclass
class ValidationReport:
    """전체 검증 보고서"""
    results: List[ValidationResult] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """에러가 없으면 유효"""
        return not any(
            r.level == ValidationLevel.ERROR and not r.is_valid
            for r in self.results
        )

    @property
    def error_count(self) -> int:
        return sum(1 for r in self.results
                   if r.level == ValidationLevel.ERROR and not r.is_valid)

    @property
    def warning_count(self) -> int:
        return sum(1 for r in self.results
                   if r.level == ValidationLevel.WARNING and not r.is_valid)

    def add_result(self, result: ValidationResult):
        self.results.append(result)

    def get_errors(self) -> List[ValidationResult]:
        return [r for r in self.results
                if r.level == ValidationLevel.ERROR and not r.is_valid]

    def get_warnings(self) -> List[ValidationResult]:
        return [r for r in self.results
                if r.level == ValidationLevel.WARNING and not r.is_valid]

    def to_dict(self) -> Dict:
        return {
            'is_valid': self.is_valid,
            'error_count': self.error_count,
            'warning_count': self.warning_count,
            'results': [
                {
                    'valid': r.is_valid,
                    'level': r.level.value,
                    'message': r.message,
                    'details': r.details
                }
                for r in self.results
            ]
        }

    def __str__(self):
        lines = [
            f"=== 검증 보고서 ===",
            f"전체 결과: {'✓ 유효' if self.is_valid else '✗ 무효'}",
            f"오류: {self.error_count}, 경고: {self.warning_count}",
            ""
        ]

        if self.results:
            lines.append("상세 결과:")
            for r in self.results:
                status = "✓" if r.is_valid else "✗"
                lines.append(f"  {status} [{r.level.value}] {r.message}")

        return "\n".join(lines)


class LogicValidator:
    """
    가계도 논리 검증 클래스

    검증 항목:
    1. 멘델 유전 법칙 준수
    2. 성염색체 연관 유전 규칙
    3. 가족 관계 일관성
    4. DNA 상대량 표 정확성
    """

    def validate_logic(
        self,
        family: Family,
        genes: Optional[List[Gene]] = None
    ) -> ValidationReport:
        """
        전체 논리 검증 수행

        Args:
            family: 검증할 가족 객체
            genes: 검증할 유전자 목록 (None이면 가족에서 추출)

        Returns:
            ValidationReport 객체
        """
        report = ValidationReport()

        # 유전자 목록 확보
        if genes is None:
            genes = self._extract_genes_from_family(family)

        # 1. 가족 구조 검증
        structure_results = self._validate_family_structure(family)
        for r in structure_results:
            report.add_result(r)

        # 2. 각 유전자별 유전 규칙 검증
        for gene in genes:
            inheritance_results = self._validate_inheritance(family, gene)
            for r in inheritance_results:
                report.add_result(r)

        # 3. 성염색체 연관 특수 규칙 검증
        for gene in genes:
            if gene.chromosome_type == ChromosomeType.X_LINKED:
                x_linked_results = self._validate_x_linked_rules(family, gene)
                for r in x_linked_results:
                    report.add_result(r)

        # 4. DNA 상대량 일관성 검증
        dna_results = self._validate_dna_amounts(family, genes)
        for r in dna_results:
            report.add_result(r)

        return report

    def _extract_genes_from_family(self, family: Family) -> List[Gene]:
        """가족 구성원에서 유전자 정보 추출"""
        genes = {}
        for member in family.all_members:
            for symbol, trait in member.traits.items():
                if symbol not in genes:
                    genes[symbol] = trait.gene
        return list(genes.values())

    def _validate_family_structure(self, family: Family) -> List[ValidationResult]:
        """가족 구조 검증"""
        results = []

        # 모든 구성원이 적절한 세대에 있는지
        for member in family.all_members:
            if member.generation < 0 or member.generation > 2:
                results.append(ValidationResult(
                    is_valid=False,
                    level=ValidationLevel.ERROR,
                    message=f"{member.id}의 세대 값이 유효하지 않음: {member.generation}",
                    details={'person_id': member.id, 'generation': member.generation}
                ))

        # 부모-자녀 관계 일관성
        for member in family.all_members:
            if member.father_id and member.mother_id:
                father = family.get_member(member.father_id)
                mother = family.get_member(member.mother_id)

                # 부모 성별 확인
                if father and father.gender != Gender.MALE:
                    results.append(ValidationResult(
                        is_valid=False,
                        level=ValidationLevel.ERROR,
                        message=f"{member.id}의 아버지 {father.id}가 남성이 아님",
                        details={'child_id': member.id, 'father_id': father.id}
                    ))

                if mother and mother.gender != Gender.FEMALE:
                    results.append(ValidationResult(
                        is_valid=False,
                        level=ValidationLevel.ERROR,
                        message=f"{member.id}의 어머니 {mother.id}가 여성이 아님",
                        details={'child_id': member.id, 'mother_id': mother.id}
                    ))

                # 세대 관계 확인
                if father and father.generation >= member.generation:
                    results.append(ValidationResult(
                        is_valid=False,
                        level=ValidationLevel.ERROR,
                        message=f"아버지 {father.id}가 자녀 {member.id}보다 같거나 낮은 세대",
                        details={'father_gen': father.generation,
                                'child_gen': member.generation}
                    ))

        if not results:
            results.append(ValidationResult(
                is_valid=True,
                level=ValidationLevel.INFO,
                message="가족 구조 검증 통과"
            ))

        return results

    def _validate_inheritance(
        self,
        family: Family,
        gene: Gene
    ) -> List[ValidationResult]:
        """멘델 유전 법칙 검증"""
        results = []

        for member in family.all_members:
            if not member.father_id or not member.mother_id:
                continue

            father = family.get_member(member.father_id)
            mother = family.get_member(member.mother_id)

            if not father or not mother:
                continue

            # 유전 규칙 검증
            is_valid, error_msg = InheritanceRule.validate_inheritance(
                father, mother, member, gene
            )

            if not is_valid:
                results.append(ValidationResult(
                    is_valid=False,
                    level=ValidationLevel.ERROR,
                    message=f"유전 규칙 위반 ({gene.symbol}): {error_msg}",
                    details={
                        'gene': gene.symbol,
                        'child_id': member.id,
                        'father_id': father.id,
                        'mother_id': mother.id
                    }
                ))
            else:
                results.append(ValidationResult(
                    is_valid=True,
                    level=ValidationLevel.INFO,
                    message=f"{member.id}의 {gene.symbol} 유전 규칙 준수"
                ))

        return results

    def _validate_x_linked_rules(
        self,
        family: Family,
        gene: Gene
    ) -> List[ValidationResult]:
        """X염색체 연관 유전 특수 규칙 검증"""
        results = []

        for member in family.all_members:
            trait = member.get_trait(gene.symbol)
            if not trait:
                continue

            # 남성 X-linked 검증: allele2가 None이어야 함
            if member.gender == Gender.MALE:
                if trait.allele2 is not None:
                    results.append(ValidationResult(
                        is_valid=False,
                        level=ValidationLevel.ERROR,
                        message=f"남성 {member.id}가 X연관 유전자 2개를 가짐",
                        details={
                            'person_id': member.id,
                            'gene': gene.symbol,
                            'genotype': trait.genotype
                        }
                    ))

            # 여성 X-linked 검증: allele2가 있어야 함
            if member.gender == Gender.FEMALE:
                if trait.allele2 is None:
                    results.append(ValidationResult(
                        is_valid=False,
                        level=ValidationLevel.ERROR,
                        message=f"여성 {member.id}가 X연관 유전자 1개만 가짐",
                        details={
                            'person_id': member.id,
                            'gene': gene.symbol,
                            'genotype': trait.genotype
                        }
                    ))

        # 아버지-딸 규칙: 아버지의 X가 반드시 딸에게 전달
        for member in family.all_members:
            if member.gender != Gender.FEMALE:
                continue
            if not member.father_id:
                continue

            father = family.get_member(member.father_id)
            if not father:
                continue

            father_trait = father.get_trait(gene.symbol)
            daughter_trait = member.get_trait(gene.symbol)

            if father_trait and daughter_trait:
                father_allele = father_trait.allele1
                daughter_alleles = {daughter_trait.allele1, daughter_trait.allele2}

                if father_allele not in daughter_alleles:
                    results.append(ValidationResult(
                        is_valid=False,
                        level=ValidationLevel.ERROR,
                        message=(f"아버지 {father.id}의 X대립유전자({father_allele})가 "
                                f"딸 {member.id}에게 전달되지 않음"),
                        details={
                            'father_id': father.id,
                            'daughter_id': member.id,
                            'father_allele': father_allele,
                            'daughter_alleles': list(daughter_alleles)
                        }
                    ))

        if not results:
            results.append(ValidationResult(
                is_valid=True,
                level=ValidationLevel.INFO,
                message=f"{gene.symbol} X염색체 연관 규칙 검증 통과"
            ))

        return results

    def _validate_dna_amounts(
        self,
        family: Family,
        genes: List[Gene]
    ) -> List[ValidationResult]:
        """DNA 상대량 일관성 검증"""
        results = []

        for member in family.all_members:
            for gene in genes:
                trait = member.get_trait(gene.symbol)
                if not trait:
                    continue

                amounts = trait.get_dna_amounts()
                d_allele = gene.dominant_allele
                r_allele = gene.recessive_allele

                # 총합 검증
                total = amounts.get(d_allele, 0) + amounts.get(r_allele, 0)

                expected_total = 1 if (gene.chromosome_type == ChromosomeType.X_LINKED
                                       and member.gender == Gender.MALE) else 2

                if total != expected_total:
                    results.append(ValidationResult(
                        is_valid=False,
                        level=ValidationLevel.ERROR,
                        message=(f"{member.id}의 {gene.symbol} DNA 상대량 합이 "
                                f"잘못됨: {total} (예상: {expected_total})"),
                        details={
                            'person_id': member.id,
                            'gene': gene.symbol,
                            'amounts': amounts,
                            'expected_total': expected_total
                        }
                    ))

                # 음수 값 검증
                for allele, amount in amounts.items():
                    if amount < 0:
                        results.append(ValidationResult(
                            is_valid=False,
                            level=ValidationLevel.ERROR,
                            message=f"{member.id}의 {allele} DNA 상대량이 음수: {amount}",
                            details={'person_id': member.id, 'allele': allele}
                        ))

        if not any(not r.is_valid for r in results):
            results.append(ValidationResult(
                is_valid=True,
                level=ValidationLevel.INFO,
                message="DNA 상대량 검증 통과"
            ))

        return results

    def validate_dna_table(
        self,
        table: DNATable,
        family: Family,
        genes: List[Gene]
    ) -> ValidationReport:
        """
        DNA 상대량 표 검증

        Args:
            table: 검증할 DNA 테이블
            family: 가족 객체
            genes: 유전자 목록

        Returns:
            ValidationReport
        """
        report = ValidationReport()

        answer_data = table.to_answer_dict()

        for row_data in answer_data:
            person_id = row_data.get('person_id')
            person = family.get_member(person_id)

            if not person:
                report.add_result(ValidationResult(
                    is_valid=False,
                    level=ValidationLevel.ERROR,
                    message=f"테이블의 {person_id}가 가족에 존재하지 않음"
                ))
                continue

            # 각 대립유전자 값 검증
            for gene in genes:
                for allele in [gene.dominant_allele, gene.recessive_allele]:
                    table_value = row_data.get(allele)
                    trait = person.get_trait(gene.symbol)

                    if trait:
                        actual_amounts = trait.get_dna_amounts()
                        actual_value = actual_amounts.get(allele, 0)

                        if table_value is not None and table_value != actual_value:
                            report.add_result(ValidationResult(
                                is_valid=False,
                                level=ValidationLevel.ERROR,
                                message=(f"{person_id}의 {allele} 값 불일치: "
                                        f"테이블={table_value}, 실제={actual_value}"),
                                details={
                                    'person_id': person_id,
                                    'allele': allele,
                                    'table_value': table_value,
                                    'actual_value': actual_value
                                }
                            ))

        if report.error_count == 0:
            report.add_result(ValidationResult(
                is_valid=True,
                level=ValidationLevel.INFO,
                message="DNA 테이블 검증 통과"
            ))

        return report

    def validate_problem_solvability(
        self,
        table: DNATable,
        family: Family,
        genes: List[Gene]
    ) -> ValidationReport:
        """
        문제 풀이 가능성 검증

        주어진 정보만으로 문제를 풀 수 있는지 확인

        Returns:
            ValidationReport
        """
        report = ValidationReport()

        # 공개된 정보 수집
        visible_info = {}
        for row in table.rows:
            person_id = row.person_id
            visible_info[person_id] = {}
            for allele, cell in row.cells.items():
                if not cell.is_hidden and cell.value is not None:
                    visible_info[person_id][allele] = cell.value

        # 각 숨겨진 값에 대해 추론 가능성 검증
        for row in table.rows:
            for allele, cell in row.cells.items():
                if cell.is_hidden and cell.value is not None:
                    # 이 값을 추론할 수 있는지 확인
                    can_infer = self._can_infer_value(
                        row.person_id, allele, family, genes, visible_info
                    )

                    if not can_infer:
                        report.add_result(ValidationResult(
                            is_valid=False,
                            level=ValidationLevel.WARNING,
                            message=(f"{row.person_name}의 {allele} 값을 "
                                    "주어진 정보로 추론하기 어려울 수 있음"),
                            details={
                                'person_id': row.person_id,
                                'allele': allele
                            }
                        ))

        if report.warning_count == 0:
            report.add_result(ValidationResult(
                is_valid=True,
                level=ValidationLevel.INFO,
                message="문제 풀이 가능성 검증 통과"
            ))

        return report

    def _can_infer_value(
        self,
        person_id: str,
        allele: str,
        family: Family,
        genes: List[Gene],
        visible_info: Dict
    ) -> bool:
        """주어진 정보로 값 추론 가능 여부"""
        # 간단한 휴리스틱:
        # 1. 같은 사람의 다른 대립유전자 값이 있으면 추론 가능
        # 2. 부모 또는 자녀의 정보가 있으면 추론 가능

        person = family.get_member(person_id)
        if not person:
            return False

        # 같은 사람의 다른 정보
        person_info = visible_info.get(person_id, {})
        if len(person_info) > 0:
            return True

        # 부모 정보
        father, mother = family.get_parents(person_id)
        if father and father.id in visible_info:
            return True
        if mother and mother.id in visible_info:
            return True

        # 자녀 정보
        children = family.get_children(person_id)
        for child in children:
            if child.id in visible_info:
                return True

        return False


def validate_logic(family: Family, genes: Optional[List[Gene]] = None) -> ValidationReport:
    """
    편의 함수: 가계도 논리 검증 수행

    Args:
        family: 검증할 가족 객체
        genes: 검증할 유전자 목록

    Returns:
        ValidationReport 객체
    """
    validator = LogicValidator()
    return validator.validate_logic(family, genes)
