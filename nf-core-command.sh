nextflow run nf-core/rnaseq -r 3.14.0 -profile docker --input $SAMPLESHEET --outdir $OUTDIT --genome GRCh37 --max_memory 254.GB --max_cpus 32 --max_time 72.h

nextflow run nf-core/rnavar -r 1.1.0 -profile docker   --input $SAMPLESHEET   --outdir $OUTDIT   --genome GRCh37   --dbsnp     Mills_and_1000G_gold_standard.indels.hg19.sites.vcf.gz    --max_memory 254.GB --max_cpus 32 --max_time 72.h