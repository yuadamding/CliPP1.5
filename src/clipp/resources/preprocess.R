# Single-sample, autosomal, 1-based inclusive intervals. Fail on ambiguous data.
args <- commandArgs(trailingOnly=TRUE)
if (length(args) != 5L) stop("Expected SNV CNA purity sample_id output_directory")
snv.file <- args[1]; cn.file <- args[2]; purity.file <- args[3]
output.prefix <- args[5]
read_input <- function(path, required) {
    widths <- count.fields(path, sep="", quote="", comment.char="")
    widths <- widths[widths > 0]
    if (length(widths) < 2L || any(widths != widths[1]))
        stop("Every input row must have exactly the header's number of fields: ", path)
    x <- read.table(path, header=TRUE, stringsAsFactors=FALSE, colClasses="character",
                    check.names=FALSE, comment.char="", quote="", na.strings=c("NA", "NaN", ""))
    if (anyDuplicated(names(x)) || !all(required %in% names(x))) stop("Missing or duplicated input columns: ", path)
    if (!nrow(x)) stop("Empty input: ", path)
    x
}
number <- function(x) suppressWarnings(as.numeric(x))
chromosome <- function(x) {
    raw <- sub("^chr", "", x, ignore.case=TRUE)
    valid <- !is.na(raw) & grepl("^[0-9]+$", raw)
    z <- number(raw)
    if (any(!valid | !is.finite(z) | !(z %in% 1:22)))
        stop("Only autosomes 1..22 or chr1..chr22 are supported; X/Y/23/24/MT and missing values are rejected")
    as.integer(z)
}
coordinate <- function(x, name) {
    z <- number(x)
    if (any(!is.finite(z) | z != floor(z) | z < 1 | z > .Machine$integer.max))
        stop(name, " must contain 1-based positive integer coordinates within int32")
    z
}
pp <- scan(purity.file, what="character", quiet=TRUE, comment.char="", quote="")
if (length(pp) != 1L) stop("Purity file must contain exactly one scalar")
purity <- number(pp)
if (!is.finite(purity) || purity <= 0 || purity > 1) stop("Purity must be in (0, 1]")
snv <- read_input(snv.file, c("chromosome_index", "position", "ref_count", "alt_count"))
cn <- read_input(cn.file, c("chromosome_index", "start_position", "end_position", "major_cn", "minor_cn", "total_cn"))
original.chrom <- snv$chromosome_index
original.pos <- snv$position
snv$chromosome_index <- chromosome(snv$chromosome_index)
snv$position <- coordinate(snv$position, "SNV position")
ids <- paste(snv$chromosome_index, snv$position, sep=":")
if (anyDuplicated(ids)) stop("Duplicate normalized SNV coordinates; legacy format permits one variant per locus")
if ("mutation_id" %in% names(snv)) {
    ids <- snv$mutation_id
    if (anyNA(ids) || any(ids == "") || anyDuplicated(ids)) stop("Mutation IDs must be nonempty and unique")
}
cn$chromosome_index <- chromosome(cn$chromosome_index)
cn$start_position <- coordinate(cn$start_position, "CNA start")
cn$end_position <- coordinate(cn$end_position, "CNA end")
if (any(cn$start_position > cn$end_position)) stop("CNA start must not exceed end (inclusive coordinates)")
for (name in c("major_cn", "minor_cn", "total_cn")) cn[[name]] <- number(cn[[name]])
cn.values <- as.matrix(cn[c("major_cn", "minor_cn", "total_cn")])
if (any(!is.finite(cn.values)) || any(cn.values != floor(cn.values)) ||
    any(cn.values > .Machine$integer.max) || any(cn$major_cn < 1) ||
    any(cn$minor_cn < 0) || any(cn$major_cn < cn$minor_cn) ||
    any(cn$total_cn != cn$major_cn + cn$minor_cn)) stop("Invalid integer major/minor/total copy numbers")
cn$segment_id <- seq_len(nrow(cn))
for (chr in unique(cn$chromosome_index)) {
    segment <- cn[cn$chromosome_index == chr, ]
    segment <- segment[order(segment$start_position, segment$end_position), ]
    if (nrow(segment) > 1L && any(segment$start_position[-1] <= head(cummax(segment$end_position), -1)))
        stop("Ambiguous overlapping CNA intervals on chromosome ", chr, "; inclusive endpoints cannot overlap")
}
if (!requireNamespace("data.table", quietly=TRUE)) stop("R package data.table is required")
snv.dt <- data.table::data.table(row_id=seq_len(nrow(snv)), chromosome_index=snv$chromosome_index, position=snv$position)
cn.dt <- data.table::as.data.table(cn)
hits <- cn.dt[snv.dt, on=.(chromosome_index, start_position <= position, end_position >= position), nomatch=0]
matched <- rep(NA_integer_, nrow(snv))
matched[hits$row_id] <- hits$segment_id
alt <- number(snv$alt_count); ref <- number(snv$ref_count); depth <- alt + ref
valid.counts <- is.finite(alt) & is.finite(ref) & alt >= 0 & ref >= 0 & depth > 0 &
                alt == floor(alt) & ref == floor(ref) & depth <= .Machine$integer.max
reason <- ifelse(!valid.counts, "invalid_counts_or_zero_depth", ifelse(is.na(matched), "no_cna_segment", "retained"))
keep <- which(reason == "retained")
ledger <- data.frame(original_row=seq_len(nrow(snv)), mutation_id=ids,
    original_chromosome=original.chrom, original_position=original.pos,
    chromosome_index=snv$chromosome_index, position=snv$position,
    matched_segment_id=matched, status=ifelse(reason == "retained", "retained", "excluded"), reason=reason)
dir.create(output.prefix, recursive=TRUE, showWarnings=FALSE)
write.table(ledger, file.path(output.prefix, "input_ledger.tsv"), sep="\t", quote=FALSE, row.names=FALSE, na="NA")
write.table(cn, file.path(output.prefix, "canonical_segments.tsv"), sep="\t", quote=FALSE, row.names=FALSE, na="NA")
if (!length(keep)) stop("No retained mutations; see input_ledger.tsv")
canonical <- data.frame(original_row=keep, mutation_id=ids[keep], chromosome_index=snv$chromosome_index[keep],
    position=snv$position[keep], ref_count=ref[keep], alt_count=alt[keep], depth=depth[keep],
    major_cn=cn$major_cn[matched[keep]], minor_cn=cn$minor_cn[matched[keep]], total_cn=cn$total_cn[matched[keep]],
    matched_segment_id=matched[keep])
write.table(canonical, file.path(output.prefix, "retained.tsv"), sep="\t", quote=FALSE, row.names=FALSE)
for (name in c("r", "n", "major", "total")) {
    value <- switch(name, r=canonical$alt_count, n=canonical$depth, major=canonical$major_cn, total=canonical$total_cn)
    write.table(value, file.path(output.prefix, paste0(name, ".txt")), quote=FALSE, col.names=FALSE, row.names=FALSE)
}
index <- canonical[c("chromosome_index", "position", "total_cn", "major_cn")]
write.table(index, file.path(output.prefix, "multiplicity.txt"), quote=FALSE, col.names=FALSE, row.names=FALSE)
write.table(purity, file.path(output.prefix, "purity_ploidy.txt"), quote=FALSE, col.names=FALSE, row.names=FALSE)
write.table(ledger[ledger$status == "excluded", ], file.path(output.prefix, "excluded_SNVs.txt"), sep="\t", quote=FALSE, row.names=FALSE)
writeLines("uniform_1_to_major_v1", file.path(output.prefix, "multiplicity_model.txt"))
cat("Input rows:", nrow(snv), "Retained:", length(keep), "Excluded:", nrow(snv)-length(keep), "\n")
