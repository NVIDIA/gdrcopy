/*
 * Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in 
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <string>
#include <vector>

#define GDRCOPY_TEST(name)                                                      \
    class GDRCopy_Test_##name : public gdrcopy::testsuite::Test                 \
    {                                                                           \
        public:                                                                 \
            GDRCopy_Test_##name() : gdrcopy::testsuite::Test(#name) {}          \
            virtual ~GDRCopy_Test_##name() {}                                   \
            virtual void test();                                                \
    };                                                                          \
    GDRCopy_Test_##name gdrcopy_test_##name;                                    \
    void GDRCopy_Test_##name::test()

#define GDRCOPY_EXTENDED_TEST(name)                                             \
    class GDRCopy_Test_##name : public gdrcopy::testsuite::Test                 \
    {                                                                           \
        public:                                                                 \
            GDRCopy_Test_##name() : gdrcopy::testsuite::Test(#name, true) {}    \
            virtual ~GDRCopy_Test_##name() {}                                   \
            virtual void test();                                                \
    };                                                                          \
    GDRCopy_Test_##name gdrcopy_test_##name;                                    \
    void GDRCopy_Test_##name::test()

namespace gdrcopy {
    namespace testsuite {
        typedef enum {
            TEST_STATUS_SUCCESS = 0,
            TEST_STATUS_FAILED,
            TEST_STATUS_WAIVED
        } test_status_t;

        class Test 
        {
            public: 
                Test(std::string name, bool is_extended_test = false);
                virtual ~Test() {}
                virtual test_status_t run();

                const std::string &name() const 
                {
                    return t_name;
                }
            
            protected:
                std::string t_name;
                bool is_extended_test;

                virtual void register_test();
                virtual void test() {};

            private:
                void run_executor(int read_fd, int write_fd);
        };

        typedef std::map<std::string, Test *> TestMap;
        TestMap *get_test_map();

        /**
         * Fill `names` with all available test names.
         */
        void get_all_test_names(std::vector<std::string>& names);

        /**
         * Fill `names` with all available extended test names.
         */
        void get_all_extended_test_names(std::vector<std::string>& names);

        /**
         * Run all tests specified by name in `tests`.
         */
        int run_tests(bool print_summary, std::vector<std::string> tests);

        /**
         * Run all available tests.
         */
        int run_all_tests(bool print_summary, bool enable_extended_tests);
    }
}
