library(Matrix)
library(dplyr)
library(matrixStats)
library(BiocNeighbors)

l2norm <- function(X, by="column")
{
  if (by=="column") {
    l2norm <- sqrt(Matrix::colSums(X^2))
    if (!any(l2norm==0)) {
      X=sweep(X, 2, l2norm, "/", check.margin=FALSE)
    }
    else{
      warning("L2 norms of zero detected for distance='Cosine, no transformation")
    }
    X = X 
  } else {
    l2norm <- sqrt(Matrix::rowSums(X^2))
    if (!any(l2norm==0)) {      
      X= X/l2norm
    }
    else{
      warning("L2 norms of zero detected for distance='Cosine'")
      X = X/ pmax(l2norm,1)
    }
  }
}

#' Get KNN
#'
#' @param dat 
#' @param ref.dat 
#' @param k 
#' @param method 
#' @param dim 
#'
#' @return
#' @export
#'
#' @examples
get_knn <- function(dat, ref.dat, k, method ="cor", dim=NULL,index=NULL, build.index=FALSE, transposed=TRUE, return.distance=FALSE)
  {
    if(transposed){
      cell.id = colnames(dat)
    }
    else{
      cell.id= row.names(dat)
    }    
    if(transposed){
      if(is.null(index)){
        ref.dat = Matrix::t(ref.dat)
      }
      dat = Matrix::t(dat)
    }
    if(method=="RANN"){
      library(RANN)
      knn.result = RANN::nn2(ref.dat, dat, k=k)
    }
    else if(method %in% c("Annoy.Euclidean", "Annoy.Cosine","cor")){
      library(BiocNeighbors)      
      if(is.null(index)){
        if(method=="cor"){
          ref.dat = ref.dat - rowMeans(ref.dat)
          ref.dat = l2norm(ref.dat,by = "row")
        }
        if (method=="Annoy.Cosine"){
          ref.dat = l2norm(ref.dat,by = "row")
        }
        if(build.index){
          index= buildAnnoy(ref.dat)
        }
      }
      if (method=="Annoy.Cosine"){
        dat = l2norm(dat,by="row")
      }
      if (method=="cor"){
        dat = dat - rowMeans(dat)
        dat = l2norm(dat,by = "row")
      }
      knn.result = queryAnnoy(X= ref.dat, query=dat, k=k, precomputed = index)
    }
    else{
      stop(paste(method, "method unknown"))
    }    
    knn.index= knn.result[[1]]
    knn.distance = knn.result[[2]]
    row.names(knn.index) = row.names(knn.distance)=cell.id
    if(!return.distance){
      return(knn.index)
    }
    else{
      list(index=knn.index, distance=knn.distance)
    }
  }


#' get knn batch
#'
#' @param dat 
#' @param ref.dat 
#' @param k 
#' @param method 
#' @param dim 
#' @param batch.size 
#' @param mc.cores 
#'
#' @return
#' @export
#'
#' @examples
get_knn_batch <- function(dat, ref.dat, k, method="cor", dim=NULL, batch.size, mc.cores=1,return.distance=FALSE,...)
  {
    if(return.distance){
      fun = "knn_combine"
    }
    else{
      fun = "rbind"
    }
    results <- batch_process(x=1:ncol(dat), batch.size=batch.size, mc.cores=mc.cores, .combine=fun, FUN=function(bin){
      get_knn(dat=dat[row.names(ref.dat),bin,drop=F], ref.dat=ref.dat, k=k, method=method, dim=dim,return.distance=return.distance, ...)
    })
    return(results)
  }

knn_combine <- function(result.1, result.2)
{
  knn.index = rbind(result.1[[1]], result.2[[1]])
  knn.distance = rbind(result.1[[2]], result.2[[2]])
  return(list(knn.index, knn.distance))
}

#' Batch process
#'
#' @param x 
#' @param batch.size 
#' @param FUN 
#' @param mc.cores 
#' @param .combine 
#' @param ... 
#'
#' @return
#' @export
#'
#' @examples
batch_process <- function(x, batch.size, FUN, mc.cores=1, .combine="c",...)
  {
    require(foreach)
    require(doMC)
    if (mc.cores == 1) {
      registerDoSEQ()
    }
    else {
      registerDoMC(cores=mc.cores)
      #on.exit(parallel::stopCluster(), add = TRUE)
    }
    bins = split(x, floor((1:length(x))/batch.size))
    results= foreach(i=1:length(bins), .combine=.combine) %dopar% FUN(bins[[i]],...)
    return(results)
  }


build_train_index <- function(cl.dat, method= c("Annoy.Cosine","cor","Annoy.Euclidean"),fn=tempfile(fileext=".idx"))
  {
    library(BiocNeighbors)
    method = method[1]
    ref.dat = Matrix::t(cl.dat)
    if(method=="cor"){
      ref.dat = ref.dat - rowMeans(ref.dat)
      ref.dat = l2norm(ref.dat,by = "row")
    }
    if (method=="Annoy.Cosine"){
      ref.dat = l2norm(ref.dat,by = "row")
    }
    index= buildAnnoy(ref.dat, fname=fn)
    return(index)    
  }

build_train_index_bs <- function(cl.dat, method= c("Annoy.Cosine","cor","Annoy.Euclidean"),sample.markers.prop=0.8, iter=100, mc.cores=10,fn=tempfile(fileext=".idx"))
  {
    library(BiocNeighbors)
    require(doMC)
    require(foreach)
    registerDoMC(cores=mc.cores)
    ###for each cluster, find markers that discriminate it from other types
    train.dat <- foreach(i=1:iter, .combine="c") %dopar% {
      train.markers = sample(row.names(cl.dat), round(nrow(cl.dat) * sample.markers.prop))
      train.cl.dat = cl.dat[train.markers,]
      index = build_train_index(cl.dat = train.cl.dat, method=method, fn = paste0(fn, ".",i))
      return(list(list(cl.dat=train.cl.dat, index=index)))
    }   
  }

map_cells_knn <- function(test.dat, cl.dat, train.index=NULL, method = c("Annoy.Cosine","cor"), batch.size=5000, mc.cores=1)
  {
    cl.knn = get_knn_batch(test.dat, cl.dat, k=1, index=train.index, method=method, transposed=TRUE, batch.size=batch.size, mc.cores=mc.cores,return.distance=TRUE)
    knn.index = cl.knn[[1]]
    knn.dist = cl.knn[[2]]
    map.df = data.frame(sample_id=colnames(test.dat), cl = colnames(cl.dat)[knn.index], dist = knn.dist)
    return(map.df)
  }

map_cells_knn_big <- function(big.dat, cl.dat, select.cells, train.index=NULL, method = c("Annoy.Cosine","cor"), batch.size=10000, mc.cores=10)
  {
    library(scrattch.bigcat)
    cl.knn =  get_knn_batch_big(big.dat, cl.dat, select.cells=select.cells, k=1, index=train.index, method=method, transposed=TRUE, batch.size=batch.size, mc.cores=mc.cores,return.distance=TRUE)
    knn.index = cl.knn[[1]]
    knn.dist = cl.knn[[2]]
    map.df = data.frame(sample_id=select.cells, cl = colnames(cl.dat)[knn.index], dist = knn.dist)
    return(map.df)
  }

map_cells_knn_bs <- function(test.dat, iter=100,cl.dat=NULL,train.index.bs=NULL, method = c("Annoy.Cosine","cor"), mc.cores=20, ...)
  {
    require(doMC)
    require(foreach)
    mc.cores = min(mc.cores, length(train.index.bs))
    registerDoMC(cores=mc.cores)
    ###for each cluster, find markers that discriminate it from other types
    if(!is.null(train.index.bs)){
      iter = length(train.index.bs)
    }
    else{
      index.bs = build_train_index_bs(cl.dat, method=method,iter=iter, ...)
    }
    library(data.table)
    map.list <- foreach(i=1:iter, .combine="c") %dopar% {
      train.index = train.index.bs[[i]]$index
      cl.dat = train.index.bs[[i]]$cl.dat
      map.df=map_cells_knn(test.dat, cl.dat, train.index, method = c("Annoy.Cosine","cor"))
      map.df = list(map.df)
    }
    map.df = rbindlist(map.list)
    map.df = map.df %>% group_by(sample_id, cl) %>% summarize(freq=n(),dist = mean(dist))
    map.df$freq = map.df$freq/iter
    best.map.df = map.df %>% group_by(sample_id) %>% summarize(best.cl= cl[which.max(freq)],prob=max(freq), avg.dist = dist[which.max(freq)])
    if(method=="cor"){
      best.map.df = best.map.df%>% mutate(avg.cor = 1 - avg.dist^2/2)
    }
    return(list(map.freq=map.df, best.map.df = best.map.df))    
  }

#cl.list is cluster membership at different levels, finest at the beginning.
#val is a vector associated with sample_id
#compute z_score aggregate at different levels of clustering, start with finest level of clustering, and resort to higher level if not enough sample size
z_score <- function(cl.list, val, min.samples =100)
  {
    sample_id = names(cl.list[[1]])
    z.score = c()
    for(i in 1:length(cl.list)){
      cl=cl.list[[i]][sample_id]
      cl.size = table(cl)
      if(i !=length(cl.list)){
        select.cl = names(cl.size)[cl.size > min.samples]
      }
      else{
        select.cl = names(cl.size)
      }
      df = data.frame(sample_id = names(cl),cl=cl, val=val[names(cl)])      
      df = df %>% filter(cl %in% select.cl) %>% group_by(cl) %>% mutate(z = (val - mean(val))/sd(val))
      z.score[df$sample_id] = df$z
      sample_id = setdiff(sample_id, df$sample_id)      
    }
    return(z.score)
  }
