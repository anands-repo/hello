#include <cstdlib>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/suite/indexing/map_indexing_suite.hpp>
#include <boost/log/common.hpp>
#include <boost/log/sinks.hpp>
#include <boost/log/sources/logger.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/core/null_deleter.hpp>
// #include "Allele.h"
// #include "Callability.h"
// #include "CallabilityAnalyzer.h"
// #include "KmerMaps.h"
// #include "Graph.h"
// #include "GraphAdvanced.h"
// #include "ContigAdvanced.h"
// #include "IterativeAssembler.hh"
// #include "IterativeAssemblerOptions.hh"
// #include "AssemblyReadInfo.hh"
// #include "AssembledContig.hh"
// #include "SearchRepeats.h"
// #include "ProductConvolver.h"
// #include "ContigSearcher.h"
// #include "AlleleSearcher.h"
// #include "AlleleSearcherLite.h"
#include "AlleleSearcherLiteFiltered.h"

using namespace boost::python;
using namespace std;

// Boost logging initialization
using namespace boost::log;
typedef sinks::synchronous_sink<sinks::text_ostream_backend> text_sink;

bool enable_debug(const attribute_value_set &set)
{
    return set["Severity"].extract<int>() >= 0;
}

bool disable_debug(const attribute_value_set &set)
{
    return set["Severity"].extract<int>() > 0;
}

// Initialize boost logging facility
void initLogging(bool debug)
{
    boost::shared_ptr<text_sink> sink = boost::make_shared<text_sink>();
    boost::shared_ptr<ostream> stream(&clog, boost::null_deleter{});
    sink->locked_backend()->add_stream(stream);
    sink->set_filter(debug ? &enable_debug : &disable_debug);
    core::get()->add_sink(sink);
}

BOOST_PYTHON_MODULE(libCallability)
{
   boost::python::numpy::initialize();

   def("initLogging", &initLogging);
   
   // def("searchRepeats", searchRepeats);

   // Allow boost to handle conversion from string vector to python list
   class_<vector<string> >("stringVector")
       .def(vector_indexing_suite<vector<string> >());

   // Allow boost to handle conversion from float vector to python list
   class_<vector<float> >("floatVector")
       .def(vector_indexing_suite<vector<float> >());

   // Allow boost to handle conversion from pair
   class_<pair<size_t,size_t> >("SizetPair")
       .def_readwrite("first", &pair<size_t,size_t>::first)
       .def_readwrite("second", &pair<size_t,size_t>::second);

   // Allow string, int to be paired
   class_<pair<string,size_t> >("stringSizetPair")
       .def_readwrite("first", &pair<string,size_t>::first)
       .def_readwrite("second", &pair<string,size_t>::second);

   // Allow boost to handle conversion from vector<size_t>
   class_<vector<size_t > >("sizetVector")
       .def(vector_indexing_suite<vector<size_t > >());

   // Allow boost to handle conversion from vector<vector<float> >
   class_<vector<vector<float> > >("vectorVectorFloat")
       .def(vector_indexing_suite<vector<vector<float> > >());

   // Allow boost to handle conversion from vector<vector<size_t> >
   class_<vector<vector<size_t> > >("vectorVectorSizet")
           .def(vector_indexing_suite<vector<vector<size_t> > >());

   // Allow boost to handle conversion from vector<pair>
   class_<vector<pair<size_t,size_t> > >("pairVector")
       .def(vector_indexing_suite<vector<pair<size_t,size_t> > >());

   // Allow boost to handle pair ints
   class_<pair<int,int> >("intPair")
       .def_readwrite("first", &pair<int,int>::first)
       .def_readwrite("second", &pair<int,int>::second);

   // Allow boost to handle conversion from vector<pair>
   class_<vector<pair<int,int> > >("pairVectorInt")
       .def(vector_indexing_suite<vector<pair<int,int> > >());

   //class_<MappingObject>("MappingObject")
   //    .def_readonly("contig", &MappingObject::contig)
   //    .def_readonly("reads", &MappingObject::reads)
   //    .def_readonly("qualities", &MappingObject::qualities)
   //    .def_readonly("readRanges", &MappingObject::readRanges)
   //    .def_readwrite("logLikelihood", &MappingObject::logLikelihood)
   //    .def_readonly("readIds", &MappingObject::readIds)
   //    .def_readonly("contigId", &MappingObject::contigId)
   //    .def_readonly("contigRanges", &MappingObject::contigRanges);

   //// Allow boost to handle conversion from vector<MappingObject>
   //class_<vector<MappingObject> >("MappingVector")
   //   .def(vector_indexing_suite<vector<MappingObject> >());

   //class_<Allele>("Allele", init<const boost::python::list&, const boost::python::list&, const boost::python::list&>())
   //    .def("getPosteriorLikelihoods", &Allele::getPosteriorLikelihoods)
   //    .def("getContigLikelihoods", &Allele::getContigLikelihoods)
   //    .def("getAnalysisLikelihood", &Allele::getAnalysisLikelihood)
   //    .def("getAnalysisContigLikelihood", &Allele::getAnalysisContigLikelihood);

   //class_<Callability>("Callability", init<size_t>())
   //    .def("programFromList", &Callability::programFromList)
   //    .def("programFromReads", &Callability::programFromReads)
   //    .def("callEvidence", &Callability::callEvidence)
   //    .def("callEvidencePairwise", &Callability::callEvidencePairwise)
   //    .def("getContigLikelihood", &Callability::getContigLikelihood)
   //    .def("set_use_first_kmer_as_flag", &Callability::set_use_first_kmer_as_flag);

   //class_<CallabilityAnalyzer>("CallabilityAnalyzer", init<size_t,size_t,size_t>())
   //    .def("addReads", &CallabilityAnalyzer::addReads)
   //    .def("performAnalysis", &CallabilityAnalyzer::performAnalysis);

   //class_<Graph>("Graph", init<size_t,float>())
   //    .def("addReadsList", &Graph::addReadsList)
   //    .def("constructGraph", &Graph::constructGraph)
   //    .def("traverseGraph", &Graph::traverseGraph)
   //    .def("checkForCycles", &Graph::checkForCycles)
   //    .def("reset", &Graph::reset)
   //    .def("getEnlightenedContigs", &Graph::getEnlightenedContigs)
   //    .def("getPurgedContigs", &Graph::getPurgedContigs)
   //    .def("mapReads", &Graph::mapReads)
   //    .def_readonly("caller", &Graph::caller)
   //    .def("mappings", &Graph::mappings)
   //    .def("deleteContigs", &Graph::deleteContigs)
   //    .def("printContigs", &Graph::printContigs);

   //class_<ContigAdvanced>("ContigAdvanced")
   //    .def_readonly("contig", &ContigAdvanced::contig)
   //    .def_readonly("readIds", &ContigAdvanced::readIds);

   //// Allow boost python to translate contig vector
   //class_<vector<ContigAdvanced> >("ContigAdvancedVector")
   //    .def(vector_indexing_suite<vector<ContigAdvanced> >());

   //class_<GraphAdvanced>("GraphAdvanced", init<size_t,size_t,size_t,const boost::python::list&,const string&,const string&,bool>())
   //    .def("traverseGraph", &GraphAdvanced::traverseGraph)
   //    .def_readonly("maxK", &GraphAdvanced::maxK)
   //    .def_readonly("contigs", &GraphAdvanced::enlightenedContigs)
   //    .def("initDistribution", &GraphAdvanced::initProbabilisticTables)
   //    .def("getFeaturesAtPosition", &GraphAdvanced::getFeaturesAtPosition)
   //    .def("features", &GraphAdvanced::features)
   //    .def("isKmerCyclic", &GraphAdvanced::isKmerCyclic);

   //class_<KmerMaps>("KmerMaps", init<size_t,size_t,size_t>())
   //    .def("addReads", &KmerMaps::addReads)
   //    .def("contigFeatures", &KmerMaps::contigFeatures)
   //    .def("multiplicities", &KmerMaps::kmerMultiplicity)
   //    .def("likelihood", &KmerMaps::getContigLikelihood)
   //    .def_readonly("features", &KmerMaps::data)
   //    .def_readonly("labels", &KmerMaps::labels);

   //class_<IterativeAssemblerOptions>("IterativeAssemblerOptions")
   //    .def_readwrite("minQval", &IterativeAssemblerOptions::minQval)
   //    .def_readwrite("minlength", &IterativeAssemblerOptions::minWordLength)
   //    .def_readwrite("maxlength", &IterativeAssemblerOptions::maxWordLength);

   //class_<AssemblyReadInfo>("AssemblyReadInfo")
   //    .def_readonly("isUsed", &AssemblyReadInfo::isUsed)
   //    .def_readonly("isFiltered", &AssemblyReadInfo::isFiltered)
   //    .def_readonly("isPseudo", &AssemblyReadInfo::isPseudo)
   //    .def_readonly("contigIds", &AssemblyReadInfo::contigIds);

   //class_<AssembledContig>("AssembledContig")
   //    .def_readonly("contig", &AssembledContig::seq);

   //// Allow boost python to translate contig vector
   //class_<vector<AssembledContig> >("AssembledContigVector")
   //        .def(vector_indexing_suite<vector<AssembledContig> >());

   //class_<Features>("Features")
   //    .def_readonly("frequencies", &Features::frequencies)
   //    .def_readonly("cyclicity", &Features::cyclicity)
   //    .def_readonly("branching", &Features::branching)
   //    .def_readonly("labels", &Features::labels);

   //def("runAssembler", &runAssembler);

   //class_<ContigSearcher>("ContigSearcher", init<boost::python::list&,const size_t,const size_t>())
   //    .def("addAllele", &ContigSearcher::addAllele)
   //    .def("search", &ContigSearcher::search)
   //    .def("supportPositions", &ContigSearcher::supportPositions)
   //    .def("alleleForContig", &ContigSearcher::alleleForContig)
   //    .def("alleleInContig", &ContigSearcher::alleleInContig)
   //    .def("allelePositionInContig", &ContigSearcher::allelePositionInContig);

   //class_<AlleleSearcher>("AlleleSearcher",
   //     init<
   //         const p::list&,
   //         const p::list&,
   //         const p::list&,
   //         const p::list&,
   //         const p::list&,
   //         const string&,
   //         const size_t,
   //         const size_t,
   //         const size_t,
   //         const size_t,
   //         const size_t,
   //         const size_t,
   //         bool,
   //         bool
   //     >()
   //    )
   //    .def_readonly("refAllele", &AlleleSearcher::refAllele)
   //    .def_readonly("allelesAtSite", &AlleleSearcher::allelesAtSite)
   //    .def_readonly("differingRegions", &AlleleSearcher::differingRegions)
   //    .def_readwrite("mismatchScore", &AlleleSearcher::mismatchScore)
   //    .def_readwrite("insertScore", &AlleleSearcher::insertScore)
   //    .def_readwrite("deleteScore", &AlleleSearcher::deleteScore)
   //    .def("scoreLocations", &AlleleSearcher::scoreLocations)
   //    .def("determineDifferingRegions", &AlleleSearcher::determineDifferingRegions)
   //    .def("printMatrix", &AlleleSearcher::printMatrix)
   //    .def("computeContigs", &AlleleSearcher::computeContigs)
   //    .def("allelePositionInContig", &AlleleSearcher::allelePositionInContig)
   //    .def("supportPositions", &AlleleSearcher::supportPositions)
   //    .def("coverage", &AlleleSearcher::coverage)
   //    .def("assemble", &AlleleSearcher::assemble)
   //    .def("alleleInContig", &AlleleSearcher::alleleInContig)
   //    .def("setAlleleForAssembly", &AlleleSearcher::setAlleleForAssembly)
   //    .def("numReadsSupportingAllele", &AlleleSearcher::numReadsSupportingAllele)
   //    .def("numReadsSupportingAlleleStrict", &AlleleSearcher::numReadsSupportingAlleleStrict)
   //    .def("determineAllelesAtSite", &AlleleSearcher::determineAllelesAtSite)
   //    .def("expandRegion", &AlleleSearcher::expandRegion)
   //    .def("computeFeatures", &AlleleSearcher::computeFeatures);

   //class_<AlleleSearcherLite>("AlleleSearcherLite",
   //      init<
   //           const p::list&,
   //           const p::list&,
   //           const p::list&,
   //           const p::list&,
   //           const p::list&,
   //           const p::list&,
   //           const p::list&,
   //           const string&,
   //           size_t,
   //           size_t,
   //           bool,
   //           bool,
   //           bool,
   //           bool
   //      >()
   //    )
   //    .def_readonly("refAllele", &AlleleSearcherLite::refAllele)
   //    .def_readonly("allelesAtSite", &AlleleSearcherLite::allelesAtSite)
   //    .def_readonly("differingRegions", &AlleleSearcherLite::differingRegions)
   //    .def_readwrite("mismatchScore", &AlleleSearcherLite::mismatchScore)
   //    .def_readwrite("insertScore", &AlleleSearcherLite::insertScore)
   //    .def_readwrite("deleteScore", &AlleleSearcherLite::deleteScore)
   //    .def("scoreLocations", &AlleleSearcherLite::scoreLocations)
   //    .def("determineDifferingRegions", &AlleleSearcherLite::determineDifferingRegions)
   //    .def("coverage", &AlleleSearcherLite::coverage)
   //    .def("assemble", &AlleleSearcherLite::assemble)
   //    .def("numReadsSupportingAllele", &AlleleSearcherLite::numReadsSupportingAllele)
   //    .def("numReadsSupportingAlleleStrict", &AlleleSearcherLite::numReadsSupportingAlleleStrict)
   //    .def("determineAllelesAtSite", &AlleleSearcherLite::determineAllelesAtSite)
   //    .def("expandRegion", &AlleleSearcherLite::expandRegion)
   //    .def("computeFeatures", &AlleleSearcherLite::computeFeatures)
   //    .def("computeFeaturesAdvanced", &AlleleSearcherLite::computeFeaturesAdvanced)
   //    .def("addAlleleForAssembly", &AlleleSearcherLite::addAlleleForAssembly)
   //    .def("clearAllelesForAssembly", &AlleleSearcherLite::clearAllelesForAssembly)
   //    .def("computeFeaturesColored", &AlleleSearcherLite::computeFeaturesColored);
    
   class_<AlleleSearcherLiteFiltered>("AlleleSearcherLite",
         init<
              const p::list&,
              const p::list&,
              const p::list&,
              const p::list&,
              const p::list&,
              const p::list&,
              const p::list&,
              const p::list&,
              const string&,
              size_t,
              size_t,
              bool
         >()
       )
       .def_readonly("refAllele", &AlleleSearcherLiteFiltered::refAllele)
       .def_readonly("allelesAtSite", &AlleleSearcherLiteFiltered::allelesAtSite)
       .def_readonly("differingRegions", &AlleleSearcherLiteFiltered::differingRegions)
       .def_readwrite("mismatchScore", &AlleleSearcherLiteFiltered::mismatchScore)
       .def_readwrite("insertScore", &AlleleSearcherLiteFiltered::insertScore)
       .def_readwrite("deleteScore", &AlleleSearcherLiteFiltered::deleteScore)
       .def_readwrite("snvThreshold", &AlleleSearcherLiteFiltered::snvThreshold)
       .def_readwrite("indelThreshold", &AlleleSearcherLiteFiltered::indelThreshold)
       .def("scoreLocations", &AlleleSearcherLiteFiltered::scoreLocations)
       .def("determineDifferingRegions", &AlleleSearcherLiteFiltered::determineDifferingRegions)
       .def("coverage", &AlleleSearcherLiteFiltered::coverage)
       .def("assemble", &AlleleSearcherLiteFiltered::assemble)
       .def("numReadsSupportingAllele", &AlleleSearcherLiteFiltered::numReadsSupportingAllele)
       .def("numReadsSupportingAlleleStrict", &AlleleSearcherLiteFiltered::numReadsSupportingAlleleStrict)
       .def("determineAllelesAtSite", &AlleleSearcherLiteFiltered::determineAllelesAtSite)
       .def("expandRegion", &AlleleSearcherLiteFiltered::expandRegion)
       .def("computeFeatures", &AlleleSearcherLiteFiltered::computeFeatures)
       .def("computeFeaturesAdvanced", &AlleleSearcherLiteFiltered::computeFeaturesAdvanced)
       .def("addAlleleForAssembly", &AlleleSearcherLiteFiltered::addAlleleForAssembly)
       .def("clearAllelesForAssembly", &AlleleSearcherLiteFiltered::clearAllelesForAssembly)
       .def("computeFeaturesColored", &AlleleSearcherLiteFiltered::computeFeaturesColored)
       .def("computeFeaturesColoredSimple", &AlleleSearcherLiteFiltered::computeFeaturesColoredSimple);

   // //boost::python::def("productConvolver", productConvolverWrapper);
   // boost::python::class_<ProductConvolver>("ProductConvolver")
   //     .def("productConvolution", &ProductConvolver::productConvolution);
}
