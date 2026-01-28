"""
models.py - 데이터 모델 정의
수능 생명과학 유전 문제를 위한 핵심 클래스들
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field


class Gender(Enum):
    """성별"""
    MALE = "male"      # XY
    FEMALE = "female"  # XX


class Phenotype(Enum):
    """표현형 - 우성/열성"""
    DOMINANT = "dominant"    # 정상 (대부분의 경우)
    RECESSIVE = "recessive"  # 유전병 발현


class InheritanceMode(Enum):
    """유전 방식"""
    AUTOSOMAL_DOMINANT = "autosomal_dominant"    # 상염색체 우성
    AUTOSOMAL_RECESSIVE = "autosomal_recessive"  # 상염색체 열성
    X_LINKED_DOMINANT = "x_linked_dominant"      # X염색체 우성
    X_LINKED_RECESSIVE = "x_linked_recessive"    # X염색체 열성


@dataclass
class GeneticTrait:
    """개인의 특정 유전자에 대한 형질"""
    gene_symbol: str           # 유전자 기호 (A, B, H 등)
    genotype: str              # 유전자형 (AA, Aa, aa, XAY 등)
    phenotype: Phenotype       # 표현형

    def is_affected(self) -> bool:
        """유전병 발현 여부"""
        return self.phenotype == Phenotype.RECESSIVE


@dataclass
class Gene:
    """유전자 정보"""
    symbol: str                          # 유전자 기호
    dominant_allele: str                 # 우성 대립유전자 (A, B 등)
    recessive_allele: str                # 열성 대립유전자 (a, b 등)
    inheritance_mode: InheritanceMode    # 유전 방식
    trait_name: str = ""                 # 형질 이름 (가), (나) 등

    def get_possible_genotypes(self, gender: Gender) -> List[str]:
        """가능한 유전자형 목록 반환"""
        D, R = self.dominant_allele, self.recessive_allele

        if self.inheritance_mode in [InheritanceMode.AUTOSOMAL_DOMINANT,
                                      InheritanceMode.AUTOSOMAL_RECESSIVE]:
            return [f"{D}{D}", f"{D}{R}", f"{R}{R}"]
        else:
            # X 연관
            if gender == Gender.MALE:
                return [f"X{D}Y", f"X{R}Y"]
            else:
                return [f"X{D}X{D}", f"X{D}X{R}", f"X{R}X{R}"]


@dataclass
class Person:
    """가계도 구성원"""
    id: str                                    # 고유 ID
    gender: Gender                             # 성별
    generation: int                            # 세대 (1=조부모, 2=부모, 3=자녀)

    # 가족 관계
    father_id: Optional[str] = None            # 아버지 ID
    mother_id: Optional[str] = None            # 어머니 ID
    spouse_id: Optional[str] = None            # 배우자 ID
    children_ids: List[str] = field(default_factory=list)

    # 유전 정보
    traits: Dict[str, GeneticTrait] = field(default_factory=dict)

    # 시각화용
    display_name: str = ""                     # 표시 이름 (1, 2, 3 또는 ⓐ)
    x_pos: float = 0.0                         # X 좌표
    y_pos: float = 0.0                         # Y 좌표

    # 문제 출제용
    phenotype_hidden: bool = False             # 표현형 숨김 여부 (?로 표시)

    def get_trait(self, gene_symbol: str) -> Optional[GeneticTrait]:
        """특정 유전자에 대한 형질 반환"""
        return self.traits.get(gene_symbol)

    def set_trait(self, trait: GeneticTrait):
        """형질 설정"""
        self.traits[trait.gene_symbol] = trait

    def is_affected(self, gene_symbol: str) -> bool:
        """특정 유전병 발현 여부"""
        trait = self.get_trait(gene_symbol)
        return trait.is_affected() if trait else False

    def get_alleles(self, gene_symbol: str) -> Tuple[str, str]:
        """대립유전자 쌍 반환"""
        trait = self.get_trait(gene_symbol)
        if not trait:
            return ("", "")

        genotype = trait.genotype

        # X 연관 유전자 처리
        if genotype.startswith("X"):
            if "Y" in genotype:
                # 남성: XAY -> (XA, Y)
                allele = genotype[1]  # A 또는 a
                return (f"X{allele}", "Y")
            else:
                # 여성: XAXA, XAXa, XaXa
                allele1 = genotype[1]
                allele2 = genotype[3]
                return (f"X{allele1}", f"X{allele2}")
        else:
            # 상염색체: AA, Aa, aa
            return (genotype[0], genotype[1])


@dataclass
class Family:
    """가계도 전체 가족"""
    members: Dict[str, Person] = field(default_factory=dict)
    couples: List[Tuple[str, str]] = field(default_factory=list)  # (남편ID, 아내ID)
    genes: List[Gene] = field(default_factory=list)

    def add_member(self, person: Person):
        """구성원 추가"""
        self.members[person.id] = person

    def get_member(self, person_id: str) -> Optional[Person]:
        """구성원 조회"""
        return self.members.get(person_id)

    def add_couple(self, husband_id: str, wife_id: str):
        """부부 관계 설정"""
        husband = self.get_member(husband_id)
        wife = self.get_member(wife_id)

        if husband and wife:
            husband.spouse_id = wife_id
            wife.spouse_id = husband_id
            self.couples.append((husband_id, wife_id))

    def add_child(self, child_id: str, father_id: str, mother_id: str):
        """자녀 관계 설정"""
        child = self.get_member(child_id)
        father = self.get_member(father_id)
        mother = self.get_member(mother_id)

        if child and father and mother:
            child.father_id = father_id
            child.mother_id = mother_id
            father.children_ids.append(child_id)
            mother.children_ids.append(child_id)

    def get_generation(self, gen: int) -> List[Person]:
        """특정 세대 구성원 반환"""
        return [p for p in self.members.values() if p.generation == gen]

    def get_children(self, parent_id: str) -> List[Person]:
        """자녀 목록 반환"""
        parent = self.get_member(parent_id)
        if not parent:
            return []
        return [self.get_member(cid) for cid in parent.children_ids
                if self.get_member(cid)]

    def get_all_members_sorted(self) -> List[Person]:
        """세대순, X좌표순 정렬된 구성원 목록"""
        return sorted(self.members.values(),
                     key=lambda p: (p.generation, p.x_pos))
