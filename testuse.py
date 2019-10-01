from orf_handler_v2 import ORFHandler
from fasta_parser import Parser


def f_use():
    filename = '../RNET1/test4/lncrnas4.fasta'
    p = Parser(filename)
    p.parse()
    feat = p.gen_feat_tab()
    rscu = p.rscu_tab(False)
    # print(feat)
    # print(rscu)
    # print(len(feat), len(rscu))
    p.gen_hex_tab()


def seq_use():
    seq = 'CAGACGTCGGAATCATGTTCAGCTGCTTACCACGCCCTGTCGAAGACTGATTCTTCCAACTGCTTCCTGCAGCGAGGATGGGGGACGGGAATGTCGCAAGTCCTGTGCGGGGGAATCTAGCGCGTGCTGCTTTGTTCCGAGAACCAGCGTAGAACATTCAACAGGATATTATTTAGGACTTCATGAGCTATATGGTATCTCTTGGAAAGAAAGCCATTTACATCCAAGTACAACTGGATTTGCTGGAGTTTTGAAATTATCGCAATTATCTGAGTAAATCAGACGTTGAAAGATGGAAAGTTTGAAAGGTGGTGACAATCTTTCCAAACGTAAGATAATCGTGGGACTTGATTTTGGTACCACGTTTTCTGGGTTTGCCCACGCACATACTGACGATCCAGAGAAAGTTTACGCGAATTATGTGTATCCAGGAGGAGTAAGTAACACATACCCTAAAACCCTGACATCAAGCTTTTACGTAAAACAGGGCGAGACCGGAGTTAAGTGGCAGTTCGAAAACTGGGGATATGGTGCTCGTGAAGCGAATTCCCGAAATAATCGCAACCGAAAAGCTAAGGTACCAGCAGTATATCTTACAAAATTTAAGCTGCACCTAGCAAGCAAAGGTTTCGGGGCACCATCGGCAACACCACTTCCACAGGGGCTTACAGTGGATGTAGTAATTACCGATTATTTGCGCAGGATAGGCGAGCTAATTGTCAATATAATTCGAGATGGTTACAGTGGTGAACTGACCAAGAAGAATATCCAATGGTGTGTGACTGTTCCATCTATTTGGAACAATGATGCCAAGGCAGTCATGAAATCCTGCATGACGAATGCTGGTTTGGTGGGTGGGGTCGATGGAAGTCACCACCCGCTCATCCTGGTATTGGAACCAGAGGCTGCATCCTTTCATTGTTACAAGGTCATGAAGGAACAGATTCTCGAGGTTGGTGACAAGATCTTGGTGGCAGACATTGGGGGAGGAACCTCCGACATTGTTGTGCAGGAGGTCGTTTCTGTCGGTGAATGTTACAGAGTGAAGGAATTGACGACCAGCTCCGGGGGCTTGTGTGGCGGTAGTTACGTGGATTCTAAATTTATGGAGTTTCTACATAGAAAGATTGGACCCTGCTTGCAGAACTGCATAGACAAGTATCCCGAGGTTGAAGAGACGTTGATCAAGAATTGGGAAGTTAAAAAAACTGGATTCGGCTTATCAAGAGAATCTAGCACTTTCGTCGGCCTTCCTTCCAAATTGGCAACAGAGTGGGAGGAACAGGACGACAAAAATGGTGAATTTGGAAGGGAATACGACGAAGTAGAGATCACAGATGAGGAGATGCAATCAATTTTTGATCCCATAGTTGATCAGAATCTGGCTCTCATCGCAGACCAATTGGTTCAAGCGAATGGAGTAAAGATCATTGTTGTGGTTGGAGGATTTGCAGAATCGAAATATCTCATGGGCCGCATTAAAGCTAGATTTAAAGAGGAAGTGCCACACATCATCCACCCACCCAACCCTGCTAGTGCGGTAAGCCTTGGCGCTGTTGCATTGGCATTCAACCCTGGTACGATCGTATCTAGAGTAAGTAGAAAAACATATGGTTTTCACTGTTTGCGTAATTTTGAAAAGGGAGTTGATCCGCCCGAATATCTTGAACTTATCGACGGAGTTTCGAAATGCTATAACCGTTTCAGTGTCTACGTCAGAAAGGGTGACATTGTTTATGTGGACGATTGCATCAGCAAAACCTTTTTCCCTGGAATGCGTGGGCAGCAAAAGATACAATTACAACTTTTCAGCTCTGATGAGATTAATCCAAGATACACAGTAGGGGAGACAATTAAGAAGGAAGGTGAAATTGAAATTGACATATCGTCGGACATGAAATTGGATAAAAAACGTGAAGTGAAAGTGTCTCTCTTCTTTGGACGATCTTCCATAGAAATTAAAGCAGAGGCTATGAACTTTATTAGTTCAGGACCTCAACAATTGGAGCTTCCTGTTGTGATTGGCTACAACCCAGAAATACTTTTGGGGTTGCAGAGTGATGGAGATGCTGGGGTTAATACAGACGACCTACTCTCAGCTGGAGTTGATAAGAAAGCCGTCTACAACCTGTACAAGATAAAGCCCAACCAAATGGGTTTTTCCATCGATTCTAGAAACGGATGTGAGTTCATTGACGATGAAGCACAGGTGATTGAAGGGTTGAACCAAGTGGTGTCCCAGCTTGTTAACATGAAATTAACGAAGCAGAGGGATTCTGCTGAACAAATTGCGAAAGTTACGGATGAGGTTCTCCCAGAACCTCAGGTGGTAGTCGGTCTGGATTTCGGCACTATGCACTCTGGGTTTGCCTACGCACAGTTCAACAATCCTCAAAGGATTTACACTCATTGGGACTATCCGAATGCAGCACATGAAGCTCCATCGTGCAAAACCCTGACCAGTAACTACTATAAAAGACAATCTGGCGGGGAAGGTAGGTGGCAGCTTATGGCTTGGGGTAACAGTGCTTATGCATTGTACGCACGAGATACTCACCAGTCTAACCTTGTGAGTGATACTCTATCACACCCTACTATCGGGATGTATGTAAGTAGATTCAAGTTGCACCTGACTAGCAGAGCTACAGGGAGTTCGTCACTCGCTCCACCACTTCCACCAGGACTTACGCTGGAAGTAGTAATTACAGATTATTTGCGTGAAATGGGTGCACTAATTCTTGAGACTTTGAAAAAACATTACGGAGATGAATTCACCAAGAAGATGATCCAATGGTGCGTGACTGTTCCATCTACTTGGGATAATGCCACCATTGTTCTAATGAAATATTGCTTGACAGCTGCCGGTTTAGTGAATGGGGTTGATGGTAGTCCTCACCCGTTCGTTGTGGTTCTGGAACCCGAGGCTATCTCCTTCAGTTGTCACAACGTCGTTTTAGAAGACAAAAACCTTGAGGCTGGTGACAAGATCTTGGTCGTTGATATAATCGAGGGAACCACTGATGTTGTGGTGCAGGAGGTTGATTCCATGGAGTTTGTTGCTCTTGATCAGAGCAAAAGAGATAAGTCTATGGATGATGTATTCCGAGTGAAAGAAGTGACGAGTCATTCGGAGCCTTTATGCTTCTACTCATACTTGAACGCTGGATTCATGGAGTTTCTGCACAAGGCTATTGGACCCTGCTTGGGTGTGTGCTTAAACCACCATCCTAAAATTTTTGCCCAGTTGACTAGATTACTTGAGCCGTTGATGTCTACCTTCGGTGACTGGACTACAGATGGAGATAGCACAGAAATTTCTCTCCCAAAAATATTATTGAAAGAATGGGAGCGCTACGACAAGAGGGAGGGTAAGTTTGCAAGGGAGTCTTACGACGAGTTAGAGATCTCGTACGAGGAGATGCAATCCATTTTCGATCCAGTGGTAGAGCAGCTTGTAGGACTCATCGCCGAACAATTGGAAGTAGTGGGTGGTTTCAAGGTTCTAGTCTTGAGTGGCAGGCTTGCAGAAGTGCCATACGTCATGGAATACATTAGAAGAATCTTCGGGGGTCATATACCACAGATAATTTCCCCCAGTAACCCTGACAGTGCTGTATGTCAGGGTGCAGTGGCAATGGTTCTTAACCGTGGCAGTGCAGGTATGTCAAAAACATGTAGAAAGACGTATGGTTTCAAAGGCATGAAATTCTTTGAGAGAGGGGTCGACCCTGATGAGTACCTCCTGGAGATCAACGGAGTGGACAAGTGTGCCAACCATTTTCAAGTCTTTGTGAAGAAGGGTGAAACAGTGACCGTGGATTCTAGTATTAGCAAAATTTGTATGCCAACACAGTCTGGTCAGAAGAAGATGAAGGTTGAACTTGTGAGCTCCGACAAACTAGACCCCAGGTACACCATAGGAGAGACAGTTAGACTAGAAGGTGAGATTGAGCTTGACATATCAGAGGACATGAAGTTGGACCAACACCGAGCAGTGAAACTCACTGTCTACTTCCGGAAGTCGTCTTTGGAAGTCGTAGCAGAGGCTGTGAACTTTGTGAGCTCATCCAGCCCTCAGCACTTGCATCTTCCAGCTGAGGTTATGGGCTATTACTGAGCAAATTCTTCTTATTCGAGGAAATTAATGTATCAAATTTGTTTTTGTCACCGAAATTGGCCATGATATTCAGCATTGTCATATTGTCATTTAGATCTAGTAAGCACAGTGTGATCTTATGCGGCCACAGCTTATGCTCAGTTCTGACAATACTAGAAGTTAACAAAACGTCATACACGCATGCCACCATGCACAGA'
    seq_lnc = 'CGAGCGGTGATGGTGTTGTTGTTGTGCCCATCGGCATACATATCTGTGCCATGCGATCTATCAATGCACGCAATGGGTGTTGGTGGTGATGGTGCCCTTGTGGCAGGACTCAGTGCTTGGATGTGGGCTACGCTCCCGTGAACCTTGCCGTTGCATGGCTGCAAGCAACGTAGTGCTCTCCGGAAGATAAGCGAGAAGCTGATTTGGGTAAAGGGCGTGCTTTAGCGACGACAGCATGGGTGCCCTCGCCGTGATGTGCGGCTTGTCCTCAATCGCATCGCCACACACCTACAGATCAAGCGCCTTCGAAAGGTTCTGCCTGACTGTGAGCTATGAATGTCATCCCTCGGGTGTTTCAGTGGTGACCCAATTCTATTAAGCTTGATTATCGAACTGTGATAGGACTGTTGTTTTCGCCTAGTGATGTTTTTGCACTGTGTTCCTTTGGGATGTCTCGACGATGAGATGGTCGTGTCAGTTTGGTGGTTCCCTCAGAGACCGTCGTGCTTTGAGTGACCCCGGTGTGTAGCAGCTGCCTGGTTGTATTACAACCCCGCCTACGGCCCCTTGTGCTGTTTTGTAAAGGTGCTCTACCGTTGAAGAGTCTTACAGTTTCCCAGTTTAGGGTTTTTTGTTGATGTGCAACTCAATCTAGACTGCTGTTTGCGGTTATTGAGTATCAGGTGAGCCCATTCTCGCCGCCGCTTCCTGTCGAGGTAACTTCAGGTCATTTTTGTCAAGTGCTAGATCGATACGCATGCAAATTCGTGAACGATGTTTACTGGCTCTCACCATTCCTTCTAGAATTCGTTACGCTGGGATATATTTGAGGGGTTGTTGATAGTCTCATATGCTGGTGTGCCTGAGGCAGGATCCACAATCTAGATGATATCTTTGTCGAATTTGTGATCCAGATCTCAGTAGGGTATGGACAGCCATTTGATGATTGCACGGTGGTTAATTTTGAAGTGGCTAACGTGGCTAACGGTTGTGGAATACACTTACGAAGGACTTCTTCCGATGCGCGCTAACTAATTCCTGCTTCAATTTATGGCCACAGATTCTTTTTTTCTCCTCTTCGTTAACGATTAGTTTTACAGCTAAATTTAAGGGTGAGGACTATACAAGCTAGCCTTTCATTTGTATAAGCTGATCCATGAGCTAGGTCTGAACTGCCTGGATGACGTAATAATCTTTTCATGAGTTGAGATTTCTGAGGTCTATGTATGGGAAAAATAAAACAAAATAAATAAATAAATAAAAAACATCGTACTGAAAATGAATTTCATATTCGCTCAAGTCTAAAATCAAACATCTGCTCAAGACCTTCTGTTGATGATAATCCTCAATTGCTCCGATTTTGCGTTAACAAGC'
    seq_prob = 'CTGCAGTGAACGTTTCCTTGAGTTACTTACAAAATCAACTGTTCTTTTGTTTCAGTCGTGGAGTTCGCCTGAAGCTTGAGATATGGACTTACCAAATCCACCTTCTGCAAGAAATCAAGGTGTGCCCATACCCGGCATTGGAGGTATCGCTGCTTGTGATCTGTGGGTCTTCAAATCCTGCAACGAGGGATAATTGATTTACGGCTTTCTCTTGGGGTCTGGAAACTATGGGGCGC'
    orf = ORFHandler(seq)
    orf.ORF_former()
    orf.find_max_orf()
    orf.pept_former()
    orf.pept_counter()
    orf.syn_codons()
    orf.rcb_counter()
    orf.entropy_counter()
    orf.mean_orf_counter()
    orf.hexamer_cnt()
    # print(orf.SCUO, orf.entr_sum)
    # orf.fickett_count()
    # print(orf.fscore)


f_use()
# seq_use()
