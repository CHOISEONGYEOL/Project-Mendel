"""
models.py - 핵심 데이터 모델 정의
Person, Gene, GeneticTrait, Family 클래스
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Tuple, Set
import copy


class Gender(Enum):
    """성별 정의 (성염색체 기반)"""
    MALE = "XY"      # 남성
    FEMALE = "XX"    # 여성


class Phenotype(Enum):
    """표현형 정의"""
    DOMINANT = "정상"      # 우성 표현형 (정상)
    RECESSIVE = "발현"     # 열성 표현형 (유전병 발현)


class ChromosomeType(Enum):
    """염색체 유형"""
    AUTOSOMAL = "상염색체"
    X_LINKED = "X염색체 연관"


@dataclass
class Gene:
    """
    유전자 클래스
    - symbol: 유전자 기호 (예: 'H', 'T')
    - dominant_allele: 우성 대립유전자 (예: 'H')
    - recessive_allele: 열성 대립유전자 (예: 'h')
    - chromosome_type: 염색체 유형
    """
    symbol: str
    dominant_allele: str
    recessive_allele: str
    chromosome_type: ChromosomeType = ChromosomeType.AUTOSOMAL

    def __post_init__(self):
        # 우성은 대문자, 열성은 소문자로 통일
        self.dominant_allele = self.dominant_allele.upper()
        self.recessive_allele = self.recessive_allele.lower()

    @property
    def alleles(self) -> Tuple[str, str]:
        """가능한 대립유전자 반환"""
        return (self.dominant_allele, self.recessive_allele)


@dataclass
class GeneticTrait:
    """
    유전 형질 클래스 - 한 사람의 특정 유전자에 대한 유전자형 정보
    - gene: Gene 객체
    - allele1: 첫 번째 대립유전자
    - allele2: 두 번째 대립유전자 (X-linked 남성의 경우 None)
    """
    gene: Gene
    allele1: str
    allele2: Optional[str] = None  # X-linked 남성은 하나만 가짐

    @property
    def genotype(self) -> str:
        """유전자형 문자열 반환"""
        if self.allele2 is None:
            return self.allele1
        # 우성 대립유전자를 앞에 배치 (정렬)
        alleles = sorted([self.allele1, self.allele2],
                        key=lambda x: (x.islower(), x))
        return ''.join(alleles)

    @property
    def phenotype(self) -> Phenotype:
        """표현형 결정"""
        # 열성 대립유전자만 있으면 열성 표현형
        if self.allele2 is None:
            # X-linked 남성: 하나의 대립유전자로 결정
            return Phenotype.RECESSIVE if self.allele1.islower() else Phenotype.DOMINANT
        else:
            # 상염색체 또는 X-linked 여성: 하나라도 우성이면 우성 표현형
            has_dominant = self.allele1.isupper() or self.allele2.isupper()
            return Phenotype.DOMINANT if has_dominant else Phenotype.RECESSIVE

    @property
    def is_carrier(self) -> bool:
        """보인자 여부 (이형접합 + 우성 표현형)"""
        if self.allele2 is None:
            return False
        return (self.allele1 != self.allele2) and (self.phenotype == Phenotype.DOMINANT)

    def get_dna_amounts(self) -> Dict[str, int]:
        """
        DNA 상대량 반환
        각 대립유전자의 개수를 딕셔너리로 반환
        """
        amounts = {
            self.gene.dominant_allele: 0,
            self.gene.recessive_allele: 0
        }

        if self.allele1:
            key = self.gene.dominant_allele if self.allele1.isupper() else self.gene.recessive_allele
            amounts[key] += 1

        if self.allele2:
            key = self.gene.dominant_allele if self.allele2.isupper() else self.gene.recessive_allele
            amounts[key] += 1

        return amounts

    def get_possible_gametes(self) -> List[str]:
        """생성 가능한 생식세포(배우자)의 대립유전자 목록"""
        if self.allele2 is None:
            return [self.allele1]
        return list(set([self.allele1, self.allele2]))


@dataclass
class Person:
    """
    개인 클래스 - 가계도의 한 구성원
    """
    id: str
    gender: Gender
    generation: int  # 세대 (0: 조부모, 1: 부모, 2: 자녀)
    traits: Dict[str, GeneticTrait] = field(default_factory=dict)

    # 가족 관계
    father_id: Optional[str] = None
    mother_id: Optional[str] = None
    spouse_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)

    # 표시 정보
    display_name: Optional[str] = None
    position: Tuple[float, float] = (0, 0)  # 시각화용 좌표

    def __post_init__(self):
        if self.display_name is None:
            self.display_name = self.id

    def add_trait(self, trait: GeneticTrait):
        """유전 형질 추가"""
        self.traits[trait.gene.symbol] = trait

    def get_trait(self, gene_symbol: str) -> Optional[GeneticTrait]:
        """특정 유전자의 형질 반환"""
        return self.traits.get(gene_symbol)

    def get_phenotype(self, gene_symbol: str) -> Optional[Phenotype]:
        """특정 유전자의 표현형 반환"""
        trait = self.get_trait(gene_symbol)
        return trait.phenotype if trait else None

    def get_genotype(self, gene_symbol: str) -> Optional[str]:
        """특정 유전자의 유전자형 반환"""
        trait = self.get_trait(gene_symbol)
        return trait.genotype if trait else None

    def get_all_dna_amounts(self) -> Dict[str, Dict[str, int]]:
        """모든 형질의 DNA 상대량 반환"""
        return {
            symbol: trait.get_dna_amounts()
            for symbol, trait in self.traits.items()
        }

    @property
    def is_affected(self) -> bool:
        """유전병 발현 여부 (하나라도 열성 표현형이면 True)"""
        return any(
            trait.phenotype == Phenotype.RECESSIVE
            for trait in self.traits.values()
        )

    def __repr__(self):
        traits_str = ", ".join(
            f"{s}:{t.genotype}" for s, t in self.traits.items()
        )
        return f"Person({self.id}, {self.gender.name}, [{traits_str}])"


@dataclass
class Family:
    """
    가족(가계도) 클래스 - 3세대 가족 구조 관리
    """
    members: Dict[str, Person] = field(default_factory=dict)
    genes: List[Gene] = field(default_factory=list)

    # 세대별 구성원 ID
    generation_0: List[str] = field(default_factory=list)  # 조부모
    generation_1: List[str] = field(default_factory=list)  # 부모
    generation_2: List[str] = field(default_factory=list)  # 자녀

    # 부부 관계
    couples: List[Tuple[str, str]] = field(default_factory=list)

    def add_member(self, person: Person):
        """구성원 추가"""
        self.members[person.id] = person

        # 세대별 목록에 추가
        if person.generation == 0:
            if person.id not in self.generation_0:
                self.generation_0.append(person.id)
        elif person.generation == 1:
            if person.id not in self.generation_1:
                self.generation_1.append(person.id)
        elif person.generation == 2:
            if person.id not in self.generation_2:
                self.generation_2.append(person.id)

    def get_member(self, person_id: str) -> Optional[Person]:
        """ID로 구성원 조회"""
        return self.members.get(person_id)

    def add_couple(self, person1_id: str, person2_id: str):
        """부부 관계 설정"""
        p1 = self.get_member(person1_id)
        p2 = self.get_member(person2_id)

        if p1 and p2:
            p1.spouse_id = person2_id
            p2.spouse_id = person1_id
            self.couples.append((person1_id, person2_id))

    def add_child(self, child_id: str, father_id: str, mother_id: str):
        """부모-자식 관계 설정"""
        child = self.get_member(child_id)
        father = self.get_member(father_id)
        mother = self.get_member(mother_id)

        if child and father and mother:
            child.father_id = father_id
            child.mother_id = mother_id

            if child_id not in father.children_ids:
                father.children_ids.append(child_id)
            if child_id not in mother.children_ids:
                mother.children_ids.append(child_id)

    def get_parents(self, person_id: str) -> Tuple[Optional[Person], Optional[Person]]:
        """부모 반환 (아버지, 어머니)"""
        person = self.get_member(person_id)
        if not person:
            return None, None

        father = self.get_member(person.father_id) if person.father_id else None
        mother = self.get_member(person.mother_id) if person.mother_id else None
        return father, mother

    def get_children(self, person_id: str) -> List[Person]:
        """자녀 목록 반환"""
        person = self.get_member(person_id)
        if not person:
            return []

        return [
            self.get_member(child_id)
            for child_id in person.children_ids
            if self.get_member(child_id)
        ]

    def get_siblings(self, person_id: str) -> List[Person]:
        """형제자매 반환 (본인 제외)"""
        person = self.get_member(person_id)
        if not person or not person.father_id:
            return []

        father = self.get_member(person.father_id)
        if not father:
            return []

        return [
            self.get_member(child_id)
            for child_id in father.children_ids
            if child_id != person_id and self.get_member(child_id)
        ]

    def get_generation(self, gen: int) -> List[Person]:
        """특정 세대의 모든 구성원 반환"""
        if gen == 0:
            ids = self.generation_0
        elif gen == 1:
            ids = self.generation_1
        elif gen == 2:
            ids = self.generation_2
        else:
            return []

        return [self.get_member(pid) for pid in ids if self.get_member(pid)]

    @property
    def all_members(self) -> List[Person]:
        """모든 구성원 리스트"""
        return list(self.members.values())

    def copy(self) -> 'Family':
        """깊은 복사"""
        return copy.deepcopy(self)

    def __repr__(self):
        return (f"Family(members={len(self.members)}, "
                f"gen0={len(self.generation_0)}, "
                f"gen1={len(self.generation_1)}, "
                f"gen2={len(self.generation_2)})")
