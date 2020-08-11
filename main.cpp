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
       .def("determineDifferingRegions", &AlleleSearcherLiteFiltered::determineDifferingRegions)
       .def("assemble", &AlleleSearcherLiteFiltered::assemble)
       .def("numReadsSupportingAllele", &AlleleSearcherLiteFiltered::numReadsSupportingAllele)
       .def("numReadsSupportingAlleleStrict", &AlleleSearcherLiteFiltered::numReadsSupportingAlleleStrict)
       .def("determineAllelesAtSite", &AlleleSearcherLiteFiltered::determineAllelesAtSite)
       .def("expandRegion", &AlleleSearcherLiteFiltered::expandRegion)
       .def("addAlleleForAssembly", &AlleleSearcherLiteFiltered::addAlleleForAssembly)
       .def("clearAllelesForAssembly", &AlleleSearcherLiteFiltered::clearAllelesForAssembly)
       .def("computeFeaturesColoredSimple", &AlleleSearcherLiteFiltered::computeFeaturesColoredSimple);
}
