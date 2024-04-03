pacman::p_load(tidyverse,
               ggplot2,
               stringr,
               quanteda,
               quanteda.textstats,
               quanteda.textmodels,
               tidytext,
               patchwork,
               readr,
               data.table,
               writexl,
               readxl)


#### R SCRIPT FOR WORDFISH MODELS ON HOUSE #####
### Warning: Large Memory requirement 


df = read_excel('2_build/wf_set.xlsx')


corp <- corpus(df, text_field="text")

tokens_preproc = tokens(corp,
                        remove_punct = TRUE,
                        remove_numbers=TRUE,
                        remove_symbols=TRUE,
                        verbose = quanteda_options("verbose")) %>% tokens_tolower()


dfm_tokens <- dfm(tokens_preproc)

## anchor liberal to 1873 Democrats and conservation to 2016 Republicans
wordfish_anchor <- textmodel_wordfish(dfm_tokens, dir = c(32, 30))

## model with no anchor
wordfish <- textmodel_wordfish(dfm_tokens)

document_scores <- wordfish_anchor$theta
document_scores2 <- wordfish$theta

##scores indicates wordfish estimate for anchored model
##scores2 indicates wordfish estimate for unanchored model
df$score = document_scores
df$score2 = document_scores2

write_xlsx(df, "2_build/house_document_scores.xlsx")

#create pivoted df. used later in creating figure 4
df_wide_1 <- df %>% select(-text) %>%
  pivot_wider(names_from = party, values_from =c(score, score2), names_sep = "_") 

write_xlsx(df_wide_1, "2_build/house_document_scores_pivoted.xlsx")
###########################################################################################################

########### Redo same process for most recent five congresses (2007 - 2016)
df = df %>% filter(year >= 2007)

corp <- corpus(df, text_field="text")

tokens_preproc = tokens(corp,
                        remove_punct = TRUE,
                        remove_numbers=TRUE,
                        remove_symbols=TRUE,
                        verbose = quanteda_options("verbose")) %>% tokens_tolower()


dfm_tokens <- dfm(tokens_preproc)

tmod_wf <- textmodel_wordfish(dfm_tokens, dir = c(1, 10))

tmod_wf2 <- textmodel_wordfish(dfm_tokens)

document_scores <- tmod_wf$theta
document_scores2 <- tmod_wf2$theta

df$score = document_scores
df$score2 = document_scores2

write_xlsx(df, "2_build/house_document_scores_mostrecent.xlsx")

df_wide_1 <- df %>% select(-text) %>%
  pivot_wider(names_from = party, values_from =c(score, score2), names_sep = "_") 

write_xlsx(df_wide_1, "2_build/house_document_scores_mostrecentpivoted.xlsx")