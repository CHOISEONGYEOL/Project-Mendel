"""
data_table.py - DNA 상대량 표 생성기
수능 스타일의 DNA 상대량 데이터 테이블 생성
"""

import random
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field

from .models import Family, Person, Gene, Phenotype


@dataclass
class DNATableCell:
    """DNA 상대량 표의 개별 셀"""
    value: Optional[int]  # 0, 1, 2 또는 None (미공개)
    is_hidden: bool = False  # 문제용으로 숨김 처리
    display: str = ""  # 표시 문자열

    def __post_init__(self):
        if self.is_hidden or self.value is None:
            self.display = "?"
        else:
            self.display = str(self.value)


@dataclass
class DNATableRow:
    """DNA 상대량 표의 한 행 (한 사람)"""
    person_id: str
    person_name: str
    cells: Dict[str, DNATableCell] = field(default_factory=dict)

    def add_cell(self, allele: str, value: Optional[int], hidden: bool = False):
        self.cells[allele] = DNATableCell(value=value, is_hidden=hidden)

    def get_display_dict(self) -> Dict[str, str]:
        """표시용 딕셔너리 반환"""
        return {allele: cell.display for allele, cell in self.cells.items()}

    def get_value_dict(self) -> Dict[str, Optional[int]]:
        """실제 값 딕셔너리 반환"""
        return {allele: cell.value for allele, cell in self.cells.items()}


@dataclass
class DNATable:
    """DNA 상대량 표 전체"""
    rows: List[DNATableRow] = field(default_factory=list)
    allele_columns: List[str] = field(default_factory=list)
    title: str = "DNA 상대량"

    def add_row(self, row: DNATableRow):
        self.rows.append(row)

    def to_display_dict(self) -> List[Dict[str, Any]]:
        """표시용 데이터 반환"""
        result = []
        for row in self.rows:
            entry = {
                'person': row.person_name,
                'person_id': row.person_id
            }
            entry.update(row.get_display_dict())
            result.append(entry)
        return result

    def to_answer_dict(self) -> List[Dict[str, Any]]:
        """정답 데이터 반환"""
        result = []
        for row in self.rows:
            entry = {
                'person': row.person_name,
                'person_id': row.person_id
            }
            entry.update(row.get_value_dict())
            result.append(entry)
        return result

    def to_markdown(self, show_answer: bool = False) -> str:
        """마크다운 표 형식으로 변환"""
        if not self.rows:
            return ""

        # 헤더
        headers = ["구성원"] + self.allele_columns
        header_line = "| " + " | ".join(headers) + " |"
        separator = "|" + "|".join(["---"] * len(headers)) + "|"

        # 데이터 행
        data_lines = []
        for row in self.rows:
            if show_answer:
                values = row.get_value_dict()
            else:
                values = row.get_display_dict()

            cells = [row.person_name]
            for allele in self.allele_columns:
                val = values.get(allele, "")
                cells.append(str(val) if val is not None else "?")
            data_lines.append("| " + " | ".join(cells) + " |")

        return "\n".join([header_line, separator] + data_lines)


class DNATableGenerator:
    """
    DNA 상대량 표 생성기
    수능 문제 스타일의 표 데이터 생성
    """

    def __init__(self):
        self.difficulty_levels = {
            'easy': {'hidden_ratio': 0.1, 'min_visible': 3},
            'medium': {'hidden_ratio': 0.3, 'min_visible': 2},
            'hard': {'hidden_ratio': 0.5, 'min_visible': 1}
        }

    def generate_table(
        self,
        family: Family,
        genes: List[Gene],
        num_persons: int = 3,
        difficulty: str = 'medium',
        specific_persons: Optional[List[str]] = None
    ) -> DNATable:
        """
        DNA 상대량 표 생성

        Args:
            family: 가족 객체
            genes: 유전자 목록
            num_persons: 표에 포함할 인원 수
            difficulty: 난이도 ('easy', 'medium', 'hard')
            specific_persons: 특정 인원 ID 목록 (None이면 랜덤 선택)

        Returns:
            DNATable 객체
        """
        # 표에 포함할 인원 선택
        if specific_persons:
            selected_ids = specific_persons
        else:
            selected_ids = self._select_persons(family, num_persons)

        # 대립유전자 컬럼 구성
        allele_columns = []
        for gene in genes:
            allele_columns.extend([gene.dominant_allele, gene.recessive_allele])

        table = DNATable(allele_columns=allele_columns)

        # 난이도 설정
        diff_config = self.difficulty_levels.get(difficulty, self.difficulty_levels['medium'])

        # 각 인원에 대해 행 생성
        for person_id in selected_ids:
            person = family.get_member(person_id)
            if not person:
                continue

            row = DNATableRow(
                person_id=person.id,
                person_name=person.display_name or person.id
            )

            # 각 유전자에 대해 DNA 상대량 계산
            for gene in genes:
                trait = person.get_trait(gene.symbol)
                if trait:
                    amounts = trait.get_dna_amounts()

                    for allele in [gene.dominant_allele, gene.recessive_allele]:
                        value = amounts.get(allele, 0)
                        hidden = self._should_hide(diff_config)
                        row.add_cell(allele, value, hidden)
                else:
                    # 형질 정보 없음
                    for allele in [gene.dominant_allele, gene.recessive_allele]:
                        row.add_cell(allele, None, True)

            table.add_row(row)

        # 최소 공개 셀 보장
        self._ensure_minimum_visible(table, diff_config['min_visible'])

        return table

    def _select_persons(self, family: Family, num_persons: int) -> List[str]:
        """표에 포함할 인원 랜덤 선택"""
        all_members = family.all_members

        # 세대별로 최소 1명씩 포함하도록 선택
        selected = []

        # 각 세대에서 1명씩
        for gen in [0, 1, 2]:
            gen_members = family.get_generation(gen)
            if gen_members:
                selected.append(random.choice(gen_members).id)

        # 추가 인원 랜덤 선택
        remaining_ids = [m.id for m in all_members if m.id not in selected]
        additional_needed = num_persons - len(selected)

        if additional_needed > 0 and remaining_ids:
            additional = random.sample(
                remaining_ids,
                min(additional_needed, len(remaining_ids))
            )
            selected.extend(additional)

        return selected[:num_persons]

    def _should_hide(self, config: Dict) -> bool:
        """셀을 숨길지 결정"""
        return random.random() < config['hidden_ratio']

    def _ensure_minimum_visible(self, table: DNATable, min_visible: int):
        """최소 공개 셀 수 보장"""
        for row in table.rows:
            visible_count = sum(
                1 for cell in row.cells.values()
                if not cell.is_hidden and cell.value is not None
            )

            if visible_count < min_visible:
                # 숨겨진 셀 중 일부 공개
                hidden_alleles = [
                    allele for allele, cell in row.cells.items()
                    if cell.is_hidden and cell.value is not None
                ]
                random.shuffle(hidden_alleles)

                for i, allele in enumerate(hidden_alleles):
                    if visible_count >= min_visible:
                        break
                    row.cells[allele].is_hidden = False
                    row.cells[allele].display = str(row.cells[allele].value)
                    visible_count += 1

    def generate_problem_style_table(
        self,
        family: Family,
        genes: List[Gene],
        style: str = 'standard'
    ) -> Tuple[DNATable, Dict]:
        """
        수능 문제 스타일의 표 생성

        Args:
            family: 가족 객체
            genes: 유전자 목록
            style: 문제 스타일
                - 'standard': 기본 DNA 상대량
                - 'sum_based': 합 기반 문제 (예: H+h의 합)
                - 'ratio_based': 비율 기반 문제

        Returns:
            (table, metadata) 튜플
        """
        metadata = {
            'style': style,
            'genes': [g.symbol for g in genes],
            'hints': []
        }

        if style == 'standard':
            table = self.generate_table(family, genes, difficulty='medium')

        elif style == 'sum_based':
            table = self._generate_sum_based_table(family, genes)
            metadata['hints'].append("각 구성원의 대립유전자 합에 주목하세요.")

        elif style == 'ratio_based':
            table = self._generate_ratio_based_table(family, genes)
            metadata['hints'].append("대립유전자 간 비율을 확인하세요.")

        else:
            table = self.generate_table(family, genes)

        return table, metadata

    def _generate_sum_based_table(
        self,
        family: Family,
        genes: List[Gene]
    ) -> DNATable:
        """합 기반 문제용 표 생성"""
        # 부모와 자녀 일부 선택
        parents = family.get_generation(1)
        children = family.get_generation(2)

        selected = []
        if parents:
            selected.extend([p.id for p in parents])
        if children:
            selected.append(random.choice(children).id)

        table = self.generate_table(
            family, genes,
            num_persons=len(selected),
            difficulty='hard',
            specific_persons=selected
        )

        return table

    def _generate_ratio_based_table(
        self,
        family: Family,
        genes: List[Gene]
    ) -> DNATable:
        """비율 기반 문제용 표 생성"""
        # 이형접합자 포함 선택
        selected = []

        for member in family.all_members:
            for gene in genes:
                trait = member.get_trait(gene.symbol)
                if trait and trait.is_carrier:
                    selected.append(member.id)
                    break

        # 최소 3명 보장
        if len(selected) < 3:
            remaining = [m.id for m in family.all_members if m.id not in selected]
            selected.extend(random.sample(
                remaining,
                min(3 - len(selected), len(remaining))
            ))

        return self.generate_table(
            family, genes,
            num_persons=min(4, len(selected)),
            difficulty='medium',
            specific_persons=selected[:4]
        )

    def create_question_data(
        self,
        table: DNATable,
        family: Family,
        genes: List[Gene]
    ) -> Dict:
        """
        문제 데이터 패키지 생성

        Returns:
            {
                'table_display': 표시용 표 데이터,
                'table_answer': 정답 표 데이터,
                'family_info': 가족 관계 정보,
                'question_hints': 문제 힌트
            }
        """
        # 가족 관계 정보
        family_info = {
            'couples': [],
            'parent_child': []
        }

        for couple in family.couples:
            p1 = family.get_member(couple[0])
            p2 = family.get_member(couple[1])
            if p1 and p2:
                family_info['couples'].append({
                    'person1': {'id': p1.id, 'name': p1.display_name},
                    'person2': {'id': p2.id, 'name': p2.display_name}
                })

        for member in family.all_members:
            if member.children_ids:
                children_info = []
                for child_id in member.children_ids:
                    child = family.get_member(child_id)
                    if child:
                        children_info.append({
                            'id': child.id,
                            'name': child.display_name
                        })
                family_info['parent_child'].append({
                    'parent': {'id': member.id, 'name': member.display_name},
                    'children': children_info
                })

        # 문제 힌트 생성
        hints = self._generate_hints(family, genes)

        return {
            'table_display': table.to_display_dict(),
            'table_answer': table.to_answer_dict(),
            'table_markdown': table.to_markdown(show_answer=False),
            'answer_markdown': table.to_markdown(show_answer=True),
            'family_info': family_info,
            'question_hints': hints
        }

    def _generate_hints(self, family: Family, genes: List[Gene]) -> List[str]:
        """문제 힌트 생성"""
        hints = []

        for gene in genes:
            # 발병자 수
            affected_count = sum(
                1 for m in family.all_members
                if m.get_trait(gene.symbol) and
                m.get_trait(gene.symbol).phenotype == Phenotype.RECESSIVE
            )

            if affected_count > 0:
                hints.append(
                    f"{gene.symbol} 형질: 가계도 내 발병자 {affected_count}명"
                )

            # 보인자 존재 여부
            carrier_exists = any(
                m.get_trait(gene.symbol) and
                m.get_trait(gene.symbol).is_carrier
                for m in family.all_members
            )
            if carrier_exists:
                hints.append(f"{gene.symbol} 형질에 대한 보인자가 존재합니다.")

        return hints
