options(warn=-1)
# The script takes commandline arguments: input_SNV input_CNV purity_file sample_id Output_dir
args = commandArgs(trailingOnly=TRUE)
# debug use
snv.file      <- args[1]
cn.file       <- args[2]
purity.file   <- args[3]
sample.id     <- args[4]
output.prefix <- args[5]

print(snv.file)
print(cn.file)
print(purity.file)
print(sample.id)
print(output.prefix)

# if the input files do not exist then quit
if(!file.exists(snv.file)){
    stop(sprintf('The Input SNV file: %s Does not exist.', snv.file))
}
if(!file.exists(cn.file)){
    stop(sprintf('The Input CNV file: %s Does not exist.', cn.file))
}
if(!file.exists(purity.file)){
    stop(sprintf('The Input Purity file: %s Does not exist.', purity.file))
}


# if the output directory does not exist, then try to create the full path
if(!dir.exists(output.prefix)){
    dir.create(output.prefix, recursive = TRUE)
}
CombineReasons <- function(chrom, pos, ind, reason){
    res <- NULL
    if(length(ind) > 0){
        for (i in seq_along(ind)){
            res[i] <- sprintf("%d\t%d\t%s",chrom[ind[i]],pos[ind[i]],reason)
        }
    }
    return(res)
}
###

VALID.CONT <- 1
dropped.SNV <- NULL

pp.table  <- read.table(purity.file)
purity <- pp.table$V1[1]
if (length(purity) != 1 || !is.finite(purity) || purity <= 0 || purity > 1) {
    stop("Purity must be finite and in (0, 1].")
}

tmp.vcf        <- read.table(snv.file, header=T, stringsAsFactors = F)
if (!all(c("chromosome_index", "position", "ref_count", "alt_count") %in% names(tmp.vcf))) {
    stop("SNV input is missing required columns.")
}
mutation.chrom <- as.numeric(tmp.vcf$chromosome_index)
mutation.pos   <- as.numeric(tmp.vcf$position)
# take off the sex chromosome
valid.ind      <- which(!is.na(mutation.chrom))
drop.ind       <- which(is.na(mutation.chrom))
dropped.SNV    <- CombineReasons(mutation.chrom, mutation.pos, drop.ind, "The SNV is on sex chromosomes.")
if(length(valid.ind) < VALID.CONT ){
    # problem.sample <- rbind(problem.sample,c(sample.id, sprintf("Only %d mutations with more than 2 callers and not sex", length(valid.ind))))
    stop(sprintf('The sample with SNV %s has less than %d SNVs that are on non-sex chromosomes.',snv.file,VALID.CONT))
}
mutation.chrom <- mutation.chrom[valid.ind]
mutation.pos   <- mutation.pos[valid.ind]
minor.read     <- tmp.vcf$alt_count[valid.ind]
total.read     <- tmp.vcf$alt_count[valid.ind] + tmp.vcf$ref_count[valid.ind]
# Do not truncate fractional counts when passing them to the integer kernel.
ref.read <- total.read - minor.read
valid.ind <- which(is.finite(minor.read) & is.finite(ref.read) &
                   minor.read >= 0 & ref.read >= 0 & total.read > 0 &
                   minor.read == floor(minor.read) & ref.read == floor(ref.read) &
                   total.read <= .Machine$integer.max)
drop.ind       <- setdiff(seq_along(minor.read),valid.ind)
dropped.SNV    <- append(dropped.SNV,CombineReasons(mutation.chrom, mutation.pos, drop.ind, "The SNV has invalid read counts or zero depth."))
if(length(valid.ind) < VALID.CONT ){
    # problem.sample <- rbind(problem.sample,c(sample.id,sprintf("Only %d mutations with non-negative reads",length(valid.ind)) ))
    stop(sprintf('The sample with SNV %s has less than %d SNVs that have non-negative reads.',snv.file,VALID.CONT))
}
mutation.chrom <- mutation.chrom[valid.ind]
mutation.pos   <- mutation.pos[valid.ind]
minor.read     <- minor.read[valid.ind]
total.read     <- total.read[valid.ind]
No.mutations   <- length(valid.ind)

# process copy number

cn.tmp         <- read.table(cn.file, header = TRUE, stringsAsFactors = FALSE)
if (!all(c("chromosome_index", "start_position", "end_position", "major_cn", "minor_cn", "total_cn") %in% names(cn.tmp))) {
    stop("CNA input is missing required columns.")
}
cn.tmp         <- cn.tmp[which(!is.na(cn.tmp[, "minor_cn"])), , drop = FALSE]
cn.tmp$chromosome_index <- as.numeric(cn.tmp$chromosome_index)
cn.values <- as.matrix(cn.tmp[, c("major_cn", "minor_cn", "total_cn"), drop = FALSE])
if (!is.numeric(cn.values) || any(!is.finite(cn.values)) ||
    any(cn.values != floor(cn.values)) || any(cn.values > .Machine$integer.max) ||
    any(cn.tmp$major_cn < 1) || any(cn.tmp$minor_cn < 0) ||
    any(cn.tmp$major_cn < cn.tmp$minor_cn) ||
    any(cn.tmp$total_cn != cn.tmp$major_cn + cn.tmp$minor_cn)) {
    stop("Copy numbers must be integers with major_cn >= minor_cn >= 0, major_cn >= 1, and total_cn = major_cn + minor_cn.")
}
No.cnLines     <- nrow(cn.tmp)
if(No.cnLines == 0){
    stop(sprintf('The sample with SNV %s does not have valid copy number status.', snv.file))
}

if (!requireNamespace("data.table", quietly = TRUE)) {
    stop("The data.table package is required for fast CNA interval matching.")
}

snv.dt <- data.table::data.table(
    mut_id = seq_len(No.mutations),
    chromosome_index = mutation.chrom,
    position = mutation.pos
)

cn.dt <- data.table::as.data.table(cn.tmp)
cn.dt[, seg_id := .I]

hits <- cn.dt[
    snv.dt,
    on = .(
        chromosome_index,
        start_position <= position,
        end_position >= position
    ),
    nomatch = 0,
    allow.cartesian = TRUE
]

# Keep the same behavior as the original loop:
# if multiple CNA segments match one SNV, use the first matching CNA row.
if (nrow(hits) > 0) {
    data.table::setorder(hits, mut_id, seg_id)
    hits <- hits[, .SD[1], by = mut_id]
}

mut.cna.id <- rep(-1L, No.mutations)
if (nrow(hits) > 0) {
    mut.cna.id[hits$mut_id] <- hits$seg_id
}
valid.ind      <- which(mut.cna.id > 0)
drop.ind       <- setdiff(seq_along(minor.read),valid.ind)
dropped.SNV    <- append(dropped.SNV,CombineReasons(mutation.chrom, mutation.pos, drop.ind, "The SNV does not have valid copy number."))
if(length(valid.ind) < VALID.CONT ){
    stop(sprintf('The sample with SNV %s has less than %d SNVs that have valid copy number status.',snv.file,VALID.CONT))
}
mutation.chrom <- mutation.chrom[valid.ind]
mutation.pos   <- mutation.pos[valid.ind]
minor.read     <- minor.read[valid.ind]
total.read     <- total.read[valid.ind]
No.mutations   <- length(valid.ind)
mut.cna.id     <- mut.cna.id[valid.ind]
# Multiplicity is latent: every integer in 1..major_cn enters the likelihood.
# Store its upper bound, never a rounded VAF-based multiplicity call.
major.count <- cn.tmp[mut.cna.id, "major_cn"]
total.count <- cn.tmp[mut.cna.id, "total_cn"]
index <- cbind(mutation.chrom, mutation.pos, total.count, major.count)

write.table(minor.read, file.path(output.prefix, "r.txt"), quote=FALSE, col.names=FALSE, row.names=FALSE)
write.table(total.read, file.path(output.prefix, "n.txt"), quote=FALSE, col.names=FALSE, row.names=FALSE)
write.table(major.count, file.path(output.prefix, "major.txt"), quote=FALSE, col.names=FALSE, row.names=FALSE)
write.table(total.count, file.path(output.prefix, "total.txt"), quote=FALSE, col.names=FALSE, row.names=FALSE)
# The historical index filename is retained; column 4 now means major CN.
write.table(index, file.path(output.prefix, "multiplicity.txt"), quote=FALSE, col.names=FALSE, row.names=FALSE)
write.table(purity, file.path(output.prefix, "purity_ploidy.txt"), quote=FALSE, col.names=FALSE, row.names=FALSE)
write.table(dropped.SNV, file.path(output.prefix, "excluded_SNVs.txt"), quote=FALSE, col.names=FALSE, row.names=FALSE)
writeLines("uniform_1_to_major_v1", file.path(output.prefix, "multiplicity_model.txt"))
