"""
data_table.py - DNA 상대량 표 생성기
수능 스타일의 DNA 상대량 표 생성
"""

import random
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from .models import Family, Person, Gene, Gender


@dataclass
class DNATableConfig:
    """DNA 표 생성 설정"""
    hide_probability: float = 0.4  # 값을 ?로 숨길 확률
    show_sum: bool = True          # 합계 열 표시


class DNATableGenerator:
    """DNA 상대량 표 생성기"""

    def __init__(self, config: Optional[DNATableConfig] = None):
        self.config = config or DNATableConfig()

    def generate_table(
        self,
        family: Family,
        selected_members: Optional[List[str]] = None,
        hide_values: bool = True
    ) -> Dict:
        """
        DNA 상대량 표 생성

        Returns:
            {
                "headers": ["구성원", "A", "a", "B", "b", ...],
                "rows": [
                    {"member": "1", "values": {"A": 1, "a": 1, "B": 2, "b": "?"}, "sum": 4},
                    ...
                ],
                "answer_key": {...}  # 정답
            }
        """
        genes = family.genes

        # 대상 구성원 선택
        if selected_members:
            members = [family.get_member(pid) for pid in selected_members
                      if family.get_member(pid)]
        else:
            # 기본: 모든 구성원
            members = list(family.members.values())

        # 헤더 생성
        headers = ["구성원"]
        for gene in genes:
            headers.append(gene.dominant_allele)
            headers.append(gene.recessive_allele)
        if self.config.show_sum:
            headers.append("합")

        # 행 생성
        rows = []
        answer_key = {}

        for person in members:
            row_data = self._generate_row(person, genes, hide_values)
            rows.append(row_data["display"])
            answer_key[person.display_name] = row_data["answer"]

        return {
            "headers": headers,
            "rows": rows,
            "answer_key": answer_key
        }

    def _generate_row(
        self,
        person: Person,
        genes: List[Gene],
        hide_values: bool
    ) -> Dict:
        """개인의 DNA 상대량 행 생성"""

        values = {}
        answer = {}
        total = 0

        for gene in genes:
            trait = person.get_trait(gene.symbol)
            if not trait:
                continue

            # DNA 상대량 계산
            amounts = self._calculate_dna_amounts(trait.genotype, gene, person.gender)

            for allele, amount in amounts.items():
                answer[allele] = amount
                total += amount

                # 값 숨기기 결정
                if hide_values and random.random() < self.config.hide_probability:
                    values[allele] = "?"
                else:
                    values[allele] = amount

        # 합계
        if self.config.show_sum:
            if hide_values and random.random() < self.config.hide_probability:
                values["합"] = "?"
            else:
                values["합"] = total
            answer["합"] = total

        return {
            "display": {
                "member": person.display_name,
                "values": values
            },
            "answer": answer
        }

    def _calculate_dna_amounts(
        self,
        genotype: str,
        gene: Gene,
        gender: Gender
    ) -> Dict[str, int]:
        """
        유전자형에서 DNA 상대량 계산

        상염색체: AA -> A:2, a:0 / Aa -> A:1, a:1 / aa -> A:0, a:2
        X 연관 (남성): XAY -> A:1, a:0 / XaY -> A:0, a:1
        X 연관 (여성): XAXA -> A:2, a:0 / XAXa -> A:1, a:1 / XaXa -> A:0, a:2
        """
        D = gene.dominant_allele
        R = gene.recessive_allele

        amounts = {D: 0, R: 0}

        if genotype.startswith('X'):
            # X 연관 유전
            for char in genotype:
                if char.upper() == D.upper():
                    amounts[D] += 1
                elif char.lower() == R.lower():
                    amounts[R] += 1
        else:
            # 상염색체
            for char in genotype:
                if char == D:
                    amounts[D] += 1
                elif char == R:
                    amounts[R] += 1

        return amounts

    def generate_problem_table(
        self,
        family: Family,
        num_hidden: int = 3
    ) -> Dict:
        """
        문제용 표 생성 (일부 값만 숨김)

        Returns:
            {
                "table": {...},
                "hidden_positions": [(구성원, 대립유전자), ...],
                "answers": {...}
            }
        """
        # 먼저 완전한 표 생성
        full_table = self.generate_table(family, hide_values=False)

        # 숨길 위치 선택
        all_positions = []
        for row in full_table["rows"]:
            member = row["member"]
            for allele, value in row["values"].items():
                if allele != "합":
                    all_positions.append((member, allele))

        # 랜덤 선택
        hidden_positions = random.sample(
            all_positions,
            min(num_hidden, len(all_positions))
        )

        # 값 숨기기
        hidden_answers = {}
        for member, allele in hidden_positions:
            for row in full_table["rows"]:
                if row["member"] == member:
                    hidden_answers[(member, allele)] = row["values"][allele]
                    row["values"][allele] = "?"
                    break

        return {
            "table": full_table,
            "hidden_positions": hidden_positions,
            "answers": hidden_answers
        }
