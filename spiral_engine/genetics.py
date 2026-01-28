"""
genetics.py - 멘델 유전 법칙 구현
유전자형 결정, 부모-자녀 유전 규칙 검증
"""

from typing import List, Tuple, Optional, Set
from .models import Gene, Gender, Phenotype, InheritanceMode, GeneticTrait


class GeneticsEngine:
    """멘델 유전학 엔진"""

    @staticmethod
    def get_phenotype(gene: Gene, genotype: str) -> Phenotype:
        """유전자형에서 표현형 결정"""
        D = gene.dominant_allele  # 'A' (대문자)
        R = gene.recessive_allele  # 'a' (소문자)

        # X 연관 유전
        if gene.inheritance_mode in [InheritanceMode.X_LINKED_DOMINANT,
                                      InheritanceMode.X_LINKED_RECESSIVE]:
            # 남성 (XAY or XaY)
            if 'Y' in genotype:
                has_dominant = D in genotype
            else:
                # 여성
                has_dominant = D in genotype

            if gene.inheritance_mode == InheritanceMode.X_LINKED_DOMINANT:
                # 우성 유전: 우성 대립유전자 있으면 발현 (정상)
                return Phenotype.DOMINANT if has_dominant else Phenotype.RECESSIVE
            else:
                # 열성 유전: 우성 대립유전자 없어야 발현 (유전병)
                if 'Y' in genotype:
                    return Phenotype.RECESSIVE if not has_dominant else Phenotype.DOMINANT
                else:
                    # 여성: XaXa만 발현
                    return Phenotype.RECESSIVE if D not in genotype else Phenotype.DOMINANT

        # 상염색체 유전
        else:
            # 우성 대립유전자(대문자)가 있는지 확인
            has_dominant = D in genotype

            if gene.inheritance_mode == InheritanceMode.AUTOSOMAL_DOMINANT:
                return Phenotype.DOMINANT if has_dominant else Phenotype.RECESSIVE
            else:
                # 열성 유전: 열성 동형접합(aa)만 발현
                return Phenotype.RECESSIVE if not has_dominant else Phenotype.DOMINANT

    @staticmethod
    def get_possible_child_genotypes(
        gene: Gene,
        father_genotype: str,
        mother_genotype: str,
        child_gender: Gender
    ) -> List[str]:
        """부모 유전자형에서 가능한 자녀 유전자형 목록"""

        father_alleles = GeneticsEngine._extract_gametes(gene, father_genotype, Gender.MALE)
        mother_alleles = GeneticsEngine._extract_gametes(gene, mother_genotype, Gender.FEMALE)

        possible = set()

        for f_allele in father_alleles:
            for m_allele in mother_alleles:
                if gene.inheritance_mode in [InheritanceMode.X_LINKED_DOMINANT,
                                              InheritanceMode.X_LINKED_RECESSIVE]:
                    # X 연관 유전
                    if child_gender == Gender.MALE:
                        # 아들: 어머니에게서 X, 아버지에게서 Y
                        if f_allele == "Y" and m_allele.startswith("X"):
                            genotype = f"{m_allele}Y"
                            possible.add(genotype)
                    else:
                        # 딸: 양쪽에서 X
                        if f_allele.startswith("X") and m_allele.startswith("X"):
                            # 정렬 (우성이 앞)
                            alleles = sorted([f_allele, m_allele],
                                           key=lambda x: (x[1].lower(), x[1].isupper()),
                                           reverse=True)
                            genotype = "".join(alleles)
                            possible.add(genotype)
                else:
                    # 상염색체 유전
                    alleles = sorted([f_allele, m_allele],
                                   key=lambda x: x.lower() + ('0' if x.isupper() else '1'))
                    genotype = "".join(alleles)
                    possible.add(genotype)

        return list(possible)

    @staticmethod
    def _extract_gametes(gene: Gene, genotype: str, gender: Gender) -> List[str]:
        """유전자형에서 가능한 배우자(감수분열 산물) 추출"""
        D = gene.dominant_allele
        R = gene.recessive_allele

        if gene.inheritance_mode in [InheritanceMode.X_LINKED_DOMINANT,
                                      InheritanceMode.X_LINKED_RECESSIVE]:
            if gender == Gender.MALE:
                # 남성: X{allele}Y -> X{allele} 또는 Y
                if "Y" in genotype:
                    x_allele = genotype.replace("Y", "")
                    return [x_allele, "Y"]
                return []
            else:
                # 여성: X{a}X{b} -> X{a}, X{b}
                # XAXa -> XA, Xa
                alleles = []
                i = 0
                while i < len(genotype):
                    if genotype[i] == 'X' and i + 1 < len(genotype):
                        alleles.append(f"X{genotype[i+1]}")
                        i += 2
                    else:
                        i += 1
                return list(set(alleles)) if alleles else [f"X{D}", f"X{R}"]
        else:
            # 상염색체: AA -> A, A / Aa -> A, a / aa -> a, a
            return list(set(genotype))

    @staticmethod
    def is_valid_inheritance(
        gene: Gene,
        father_genotype: str,
        mother_genotype: str,
        child_genotype: str,
        child_gender: Gender
    ) -> bool:
        """자녀 유전자형이 부모로부터 유전 가능한지 검증"""
        possible = GeneticsEngine.get_possible_child_genotypes(
            gene, father_genotype, mother_genotype, child_gender
        )
        return child_genotype in possible

    @staticmethod
    def create_trait(gene: Gene, genotype: str) -> GeneticTrait:
        """유전자형으로부터 형질 객체 생성"""
        phenotype = GeneticsEngine.get_phenotype(gene, genotype)
        return GeneticTrait(
            gene_symbol=gene.symbol,
            genotype=genotype,
            phenotype=phenotype
        )


class InheritanceValidator:
    """유전 규칙 검증기"""

    def __init__(self, gene: Gene):
        self.gene = gene
        self.engine = GeneticsEngine()

    def validate_family(self, family) -> Tuple[bool, List[str]]:
        """전체 가족의 유전 일관성 검증"""
        errors = []

        for person in family.members.values():
            if person.father_id and person.mother_id:
                father = family.get_member(person.father_id)
                mother = family.get_member(person.mother_id)

                if not father or not mother:
                    continue

                father_trait = father.get_trait(self.gene.symbol)
                mother_trait = mother.get_trait(self.gene.symbol)
                child_trait = person.get_trait(self.gene.symbol)

                if not all([father_trait, mother_trait, child_trait]):
                    continue

                if not GeneticsEngine.is_valid_inheritance(
                    self.gene,
                    father_trait.genotype,
                    mother_trait.genotype,
                    child_trait.genotype,
                    person.gender
                ):
                    errors.append(
                        f"{person.display_name}: 유전자형 {child_trait.genotype}은 "
                        f"부모({father_trait.genotype} x {mother_trait.genotype})에서 불가능"
                    )

        return len(errors) == 0, errors
