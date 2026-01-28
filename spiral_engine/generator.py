"""
generator.py - 3세대 연결 가계도 생성기
수능 스타일: 두 조부모 가족이 2세대에서 결혼으로 연결되는 구조
Backtracking 알고리즘으로 유전적으로 일관된 가계도 생성
"""

import random
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass, field

from .models import Family, Person, Gender, Gene, Phenotype, InheritanceMode, GeneticTrait
from .genetics import GeneticsEngine


@dataclass
class GenerationConfig:
    """가계도 생성 설정"""
    # 세대별 구성
    num_children_left: int = 2      # 왼쪽 조부모의 자녀 수
    num_children_right: int = 2     # 오른쪽 조부모의 자녀 수
    num_children_gen3: int = 2      # 3세대 자녀 수

    ensure_both_genders: bool = True

    # 유전자 설정
    genes: List[Gene] = field(default_factory=list)

    # 유전병 발현 확률 제어
    affected_probability: float = 0.3

    # 문제 출제용: 마지막 구성원 숨기기 (?)
    hide_last_member: bool = True


class PedigreeGenerator:
    """3세대 연결 가계도 생성기 - 수능 스타일"""

    def __init__(self):
        self.engine = GeneticsEngine()

    def generate(self, config: Optional[GenerationConfig] = None) -> Family:
        """
        완전한 3세대 가계도 생성

        구조 (수능 기출 스타일):
        [□]─[○]           [□]─[○]      <- 1세대: 두 조부모 부부
             │                 │
          ┌──┴──┐           ┌──┴──┐
         [□]  [○]─────────[□]  [○]    <- 2세대: 자녀들 (가운데서 결혼)
          3    ⓐ           4    5
                    │
               ┌────┴────┐
              [□]       [○]           <- 3세대: 손자녀
               7         8
        """
        if config is None:
            config = GenerationConfig()

        # 기본 유전자 설정
        if not config.genes:
            config.genes = self._create_default_genes()

        # 최대 시도 횟수
        max_attempts = 50

        for attempt in range(max_attempts):
            # 1단계: 가족 구조 생성
            family = self._create_family_structure(config)

            # 2단계: 유전자형 할당 (Backtracking)
            success = self._assign_genotypes_backtracking(family, config.genes)

            if not success:
                continue

            # 3단계: 각 유전자에 대해 최소 1명 이상 발현자가 있는지 확인
            if self._has_affected_for_all_genes(family, config.genes):
                return family

        # 최대 시도 후에도 실패하면 마지막 결과 반환
        return family

    def _has_affected_for_all_genes(self, family: Family, genes: List[Gene]) -> bool:
        """각 유전자에 대해 최소 1명 이상 발현자가 있는지 확인"""
        for gene in genes:
            has_affected = False
            for person in family.members.values():
                trait = person.get_trait(gene.symbol)
                if trait and trait.is_affected():
                    has_affected = True
                    break
            if not has_affected:
                return False
        return True

    def _create_default_genes(self) -> List[Gene]:
        """기본 유전자 설정"""
        return [
            Gene(
                symbol="A",
                dominant_allele="A",
                recessive_allele="a",
                inheritance_mode=InheritanceMode.AUTOSOMAL_RECESSIVE,
                trait_name="(가)"
            ),
            Gene(
                symbol="B",
                dominant_allele="B",
                recessive_allele="b",
                inheritance_mode=InheritanceMode.AUTOSOMAL_RECESSIVE,
                trait_name="(나)"
            )
        ]

    def _create_family_structure(self, config: GenerationConfig) -> Family:
        """
        수능 스타일 3세대 가족 구조 생성

        번호 체계:
        - 1, 2: 왼쪽 조부모
        - 오른쪽 조부모: 번호 없음 (또는 별도)
        - 3, 4, 5...: 2세대 자녀들
        - ⓐ: 외부 배우자 (다른 가족에서 옴)
        - 7, 8...: 3세대 손자녀
        """
        family = Family()
        family.genes = config.genes

        person_id = 1
        display_num = 1

        # ===== 1세대: 왼쪽 조부모 =====
        gf_left = Person(
            id=f"P{person_id}", gender=Gender.MALE, generation=1,
            display_name=""  # 번호 없음
        )
        person_id += 1

        gm_left = Person(
            id=f"P{person_id}", gender=Gender.FEMALE, generation=1,
            display_name=""
        )
        person_id += 1

        family.add_member(gf_left)
        family.add_member(gm_left)
        family.add_couple(gf_left.id, gm_left.id)

        # ===== 1세대: 오른쪽 조부모 =====
        gf_right = Person(
            id=f"P{person_id}", gender=Gender.MALE, generation=1,
            display_name=str(display_num)
        )
        display_num += 1
        person_id += 1

        gm_right = Person(
            id=f"P{person_id}", gender=Gender.FEMALE, generation=1,
            display_name=str(display_num)
        )
        display_num += 1
        person_id += 1

        family.add_member(gf_right)
        family.add_member(gm_right)
        family.add_couple(gf_right.id, gm_right.id)

        # ===== 2세대: 왼쪽 조부모의 자녀들 =====
        left_children = []
        left_genders = self._generate_children_genders(config.num_children_left)

        for gender in left_genders:
            child = Person(
                id=f"P{person_id}",
                gender=gender,
                generation=2,
                display_name=str(display_num)
            )
            display_num += 1
            person_id += 1
            family.add_member(child)
            family.add_child(child.id, gf_left.id, gm_left.id)
            left_children.append(child)

        # ===== 2세대: 오른쪽 조부모의 자녀들 =====
        right_children = []
        right_genders = self._generate_children_genders(config.num_children_right)

        # 첫 번째 자녀는 ⓐ (외부 배우자 표시)
        for i, gender in enumerate(right_genders):
            if i == 0:
                # 왼쪽 자녀와 결혼할 사람 - ⓐ로 표시
                child = Person(
                    id=f"P{person_id}",
                    gender=gender,
                    generation=2,
                    display_name="ⓐ"
                )
            else:
                child = Person(
                    id=f"P{person_id}",
                    gender=gender,
                    generation=2,
                    display_name=str(display_num)
                )
                display_num += 1

            person_id += 1
            family.add_member(child)
            family.add_child(child.id, gf_right.id, gm_right.id)
            right_children.append(child)

        # ===== 2세대 결혼: 왼쪽 마지막 자녀와 오른쪽 첫 자녀 결혼 =====
        bride_from_left = left_children[-1]  # 왼쪽 가족의 마지막 자녀
        groom_from_right = right_children[0]  # 오른쪽 가족의 첫 자녀 (ⓐ)

        # 성별 맞추기
        if bride_from_left.gender == groom_from_right.gender:
            # 같은 성별이면 한쪽 변경
            if bride_from_left.gender == Gender.MALE:
                groom_from_right.gender = Gender.FEMALE
            else:
                groom_from_right.gender = Gender.MALE

        # 부부 연결
        if bride_from_left.gender == Gender.MALE:
            family.add_couple(bride_from_left.id, groom_from_right.id)
            father_id = bride_from_left.id
            mother_id = groom_from_right.id
        else:
            family.add_couple(groom_from_right.id, bride_from_left.id)
            father_id = groom_from_right.id
            mother_id = bride_from_left.id

        # ===== 3세대: 손자녀들 =====
        gen3_genders = self._generate_children_genders(config.num_children_gen3)
        gen3_count = len(gen3_genders)

        for i, gender in enumerate(gen3_genders):
            # 마지막 자녀는 표현형 숨김 (?)
            is_last = (i == gen3_count - 1) and config.hide_last_member

            grandchild = Person(
                id=f"P{person_id}",
                gender=gender,
                generation=3,
                display_name="?" if is_last else str(display_num),
                phenotype_hidden=is_last
            )

            if not is_last:
                display_num += 1
            person_id += 1
            family.add_member(grandchild)
            family.add_child(grandchild.id, father_id, mother_id)

        return family

    def _generate_children_genders(self, count: int) -> List[Gender]:
        """자녀 성별 목록 생성"""
        if count <= 0:
            return []

        genders = []
        if count >= 2:
            genders.append(Gender.MALE)
            genders.append(Gender.FEMALE)
            for _ in range(count - 2):
                genders.append(random.choice([Gender.MALE, Gender.FEMALE]))
        else:
            genders.append(random.choice([Gender.MALE, Gender.FEMALE]))

        random.shuffle(genders)
        return genders

    def _assign_genotypes_backtracking(
        self,
        family: Family,
        genes: List[Gene]
    ) -> bool:
        """Backtracking으로 유전자형 할당"""
        members = sorted(family.members.values(), key=lambda p: p.generation)

        for gene in genes:
            if not self._assign_gene_backtracking(family, gene, members, 0):
                return False

        return True

    def _assign_gene_backtracking(
        self,
        family: Family,
        gene: Gene,
        members: List[Person],
        index: int
    ) -> bool:
        """단일 유전자에 대해 backtracking"""
        if index >= len(members):
            return True

        person = members[index]
        possible = gene.get_possible_genotypes(person.gender)

        if person.father_id and person.mother_id:
            father = family.get_member(person.father_id)
            mother = family.get_member(person.mother_id)

            if father and mother:
                father_trait = father.get_trait(gene.symbol)
                mother_trait = mother.get_trait(gene.symbol)

                if father_trait and mother_trait:
                    possible = GeneticsEngine.get_possible_child_genotypes(
                        gene,
                        father_trait.genotype,
                        mother_trait.genotype,
                        person.gender
                    )

        # 열성 유전자형(발현자)을 우선 시도하도록 정렬
        # 열성 동형접합(aa, bb, XaY 등)을 앞으로
        def sort_key(genotype):
            # 열성 동형접합 > 이형접합 > 우성 동형접합 순서
            r = gene.recessive_allele
            if genotype in [f"{r}{r}", f"X{r}Y", f"X{r}X{r}"]:
                return 0  # 발현자 (가장 먼저 시도)
            elif r in genotype:
                return 1  # 보인자 (중간)
            else:
                return 2  # 정상 (마지막)

        # 30% 확률로 열성 우선, 70% 확률로 랜덤
        if random.random() < 0.4:
            possible = sorted(possible, key=sort_key)
        else:
            random.shuffle(possible)

        for genotype in possible:
            trait = GeneticsEngine.create_trait(gene, genotype)
            person.set_trait(trait)

            if self._assign_gene_backtracking(family, gene, members, index + 1):
                return True

            person.traits.pop(gene.symbol, None)

        return False
